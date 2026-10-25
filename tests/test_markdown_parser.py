import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from paper_agent.adapters.markdown_parser import (
    ExtractedDocument,
    PdfMarkdownParser,
    score_parse_quality,
    split_sections,
)
from paper_agent.domain.errors import FullTextParseError

try:
    import pypdf  # noqa: F401

    HAS_PYPDF = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_PYPDF = False

SECTIONED_TEXT = """SciBERT study of scientific text models

Abstract
We introduce a pipeline for scientific text. It is evaluated on three tasks.

1 Introduction
Scientific text differs from general text. We study the problem.

Method
We pretrain a model on a large corpus of papers.

Experiments
We evaluate on classification and tagging.

Conclusion
The model improves over baselines.

Limitations
Coverage of non-English text remains limited.
"""


def make_pdf(text: str) -> bytes:
    """Generate a minimal single-page PDF containing ``text`` (ASCII only)."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("ascii") if text else b"BT ET"
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length %d >>stream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


class FakeExtractor:
    name = "fake"
    version = "1.0"

    def __init__(self, text: str, page_count: int | None = 1, markdown_ready: bool = False):
        self.text = text
        self.page_count = page_count
        self.markdown_ready = markdown_ready
        self.calls = 0

    async def extract(self, pdf_path: Path) -> ExtractedDocument:
        self.calls += 1
        return ExtractedDocument(
            text=self.text, page_count=self.page_count, markdown_ready=self.markdown_ready
        )


class MarkdownParserTest(unittest.TestCase):
    def _write_pdf(self, tmp: str, content: bytes | None = None) -> Path:
        pdf_path = Path(tmp) / "source.pdf"
        pdf_path.write_bytes(content if content is not None else make_pdf("placeholder"))
        return pdf_path

    def test_parse_writes_markdown_and_quality_meta(self):
        extractor = FakeExtractor(SECTIONED_TEXT, page_count=2)
        parser = PdfMarkdownParser(extractor=extractor)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp)
            md_path, quality = asyncio.run(parser.parse(pdf_path, Path(tmp)))
            self.assertTrue(md_path.exists())
            markdown = md_path.read_text("utf-8")
            self.assertIn("# SciBERT study of scientific text models", markdown)
            self.assertIn("## Abstract", markdown)
            self.assertIn("## Method", markdown)
            self.assertGreater(quality, 0.0)
            self.assertLessEqual(quality, 1.0)
            meta = json.loads(md_path.with_suffix(".parse.json").read_text("utf-8"))
            self.assertEqual(meta["parser_name"], "fake")
            self.assertEqual(meta["quality"], quality)

    def test_parse_result_is_cached_by_pdf_hash(self):
        extractor = FakeExtractor(SECTIONED_TEXT)
        parser = PdfMarkdownParser(extractor=extractor)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp)
            first = asyncio.run(parser.parse(pdf_path, Path(tmp)))
            second = asyncio.run(parser.parse(pdf_path, Path(tmp)))
            self.assertEqual(first, second)
            self.assertEqual(extractor.calls, 1)

    def test_empty_extraction_raises_parse_error(self):
        parser = PdfMarkdownParser(extractor=FakeExtractor("", page_count=1))
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp)
            with self.assertRaises(FullTextParseError):
                asyncio.run(parser.parse(pdf_path, Path(tmp)))

    def test_quality_rewards_structure_and_length(self):
        structured = score_parse_quality(SECTIONED_TEXT, page_count=2)
        blob = score_parse_quality("just some unstructured words repeated. " * 3)
        self.assertGreater(structured, blob)
        self.assertGreaterEqual(blob, 0.0)

    def test_auto_backend_selects_mineru_only_with_key(self):
        with_key = PdfMarkdownParser(backend="auto", mineru_api_key="token")
        without_key = PdfMarkdownParser(backend="auto", mineru_api_key="")
        self.assertEqual(with_key.parser_name, "mineru")
        self.assertEqual(without_key.parser_name, "pypdf")
        with self.assertRaises(ValueError):
            PdfMarkdownParser(backend="something-else")

    def test_split_sections_maps_aliases(self):
        sections = split_sections(SECTIONED_TEXT)
        self.assertIn("abstract", sections)
        self.assertIn("method", sections)
        self.assertIn("experiments", sections)
        self.assertIn("conclusion", sections)
        self.assertIn("limitations", sections)

    def test_split_sections_handles_markdown_numbered_headings(self):
        # Regression: re-splitting generated Markdown lost "## 1 Introduction".
        text = (
            "## Abstract\nShort abstract.\n## 1 Introduction\nIntro text.\n"
            "## 3 Model Architecture\nEncoder decoder.\n## 7 Conclusion\nDone.\n"
        )
        sections = split_sections(text)
        self.assertEqual(sections["abstract"], "Short abstract.")
        self.assertEqual(sections["introduction"], "Intro text.")
        self.assertEqual(sections["method"], "Encoder decoder.")
        self.assertEqual(sections["conclusion"], "Done.")

    def test_split_sections_repairs_small_caps_headings(self):
        # Regression: pypdf extracts small-caps headings as "I NTRODUCTION".
        text = "1 I NTRODUCTION\nIntro.\n4 E XPERIMENT SETTINGS\nSetup.\n6 C ONCLUSION\nEnd.\n"
        sections = split_sections(text)
        self.assertEqual(sections["introduction"], "Intro.")
        self.assertEqual(sections["experiments"], "Setup.")
        self.assertEqual(sections["conclusion"], "End.")

    def test_markdown_is_written_directly_into_destination_dir(self):
        parser = PdfMarkdownParser(extractor=FakeExtractor(SECTIONED_TEXT, page_count=2))
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp)
            destination = Path(tmp) / "md-out"
            md_path, _ = asyncio.run(parser.parse(pdf_path, destination))
            self.assertEqual(md_path.parent, destination)

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_pypdf_extracts_real_pdf(self):
        # The body must exceed the parser's minimum-character threshold.
        long_text = (
            "Hello pypdf extraction test with a body long enough to pass "
            "the minimum character threshold of the parser."
        )
        parser = PdfMarkdownParser(backend="pypdf")
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp, make_pdf(long_text))
            md_path, quality = asyncio.run(parser.parse(pdf_path, Path(tmp)))
            self.assertIn("pypdf extraction", md_path.read_text("utf-8"))
            self.assertGreater(quality, 0.0)

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_blank_pdf_yields_parse_error(self):
        parser = PdfMarkdownParser(backend="pypdf")
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._write_pdf(tmp, make_pdf(""))
            with self.assertRaises(FullTextParseError):
                asyncio.run(parser.parse(pdf_path, Path(tmp)))


if __name__ == "__main__":
    unittest.main()
