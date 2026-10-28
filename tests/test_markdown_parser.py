import asyncio
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import httpx
import pytest

from paper_agent.adapters.markdown_parser import (
    ExtractedDocument,
    PdfMarkdownParser,
    _text_to_markdown,
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


def test_hard_wraps_reflow_without_destroying_structure():
    text = (
        "A scientific paper\n\nAbstract\nThis para-\ngraph is split\n"
        "over several lines.\n\nA second paragraph.\n\nMethod\n"
        "- first item\n- second item\n\n| A | B |\n| --- | --- |\n"
        "\n中文段落被\n人为换行。\n\n```python\nx = 1\ny = 2\n```\n"
        "\n$$\nx = y + z\n$$\n"
    )
    markdown = _text_to_markdown(text, "fallback")
    assert "This paragraph is split over several lines." in markdown
    assert "lines.\n\nA second paragraph." in markdown
    assert "## Abstract" in markdown and "## Method" in markdown
    assert "- first item\n- second item" in markdown
    assert "| A | B |\n| --- | --- |" in markdown
    assert "中文段落被人为换行。" in markdown
    assert "```python\nx = 1\ny = 2\n```" in markdown
    assert "$$\nx = y + z\n$$" in markdown


def _mineru_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _mineru_transport(files, requests=None):
    payload = _mineru_zip(files)

    def respond(request):
        if requests is not None:
            requests.append(request)
        if request.url.path.endswith("file-urls/batch"):
            body = json.loads(request.content)
            assert body["enable_formula"] and body["enable_table"]
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"batch_id": "test", "file_urls": ["https://upload.example/file"]},
                },
            )
        if request.method == "PUT":
            assert "authorization" not in request.headers
            return httpx.Response(200)
        if "extract-results" in request.url.path:
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(
                200,
                json={
                    "data": {
                        "extract_result": [
                            {"state": "done", "full_zip_url": "https://cdn.example/result.zip"}
                        ]
                    }
                },
            )
        assert "authorization" not in request.headers
        return httpx.Response(200, content=payload)

    return httpx.MockTransport(respond)


def test_mineru_preserves_math_saves_images_and_does_not_leak_token(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("A source paper"))
    text = (
        "# Paper\n\nScientific document with enough characters for extraction.\n\n"
        "\\[ E = mc^2 \\]\nInline \\(x+y\\).\n![figure](images/fig.png)\n"
        "```latex\n\\[do not rewrite\\]\n```\n"
    )
    requests = []
    parser = PdfMarkdownParser(
        backend="mineru",
        mineru_api_key="test-token",
        transport=_mineru_transport(
            {
                "result/full.md": text,
                "result/images/fig.png": b"image-bytes",
                "result/../evil.png": b"must-not-save",
                "result/images/secret.txt": b"not-an-image",
            },
            requests,
        ),
    )
    path, _ = asyncio.run(parser.parse(pdf, tmp_path / "md"))
    markdown = path.read_text("utf-8")
    assert "$$\nE = mc^2\n$$" in markdown and "$x+y$" in markdown
    assert "```latex\n\\[do not rewrite\\]\n```" in markdown
    assert "paper_assets/images/fig.png" in markdown
    assert (path.parent / "paper_assets/images/fig.png").read_bytes() == b"image-bytes"
    assert not (tmp_path / "evil.png").exists()
    assert not list(path.parent.rglob("*.txt"))
    meta = json.loads(path.with_suffix(".parse.json").read_text())
    assert len(meta["assets"]) == 1
    assert len(requests) == 4


def test_mineru_image_path_with_spaces_remains_readable_in_exports(tmp_path):
    from paper_agent.domain.models import Paper
    from paper_agent.ui.markdown_exports import embed_paper_images, paper_markdown_files

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("Source paper"))
    parser = PdfMarkdownParser(
        backend="mineru",
        mineru_api_key="test-token",
        transport=_mineru_transport(
            {
                "full.md": SECTIONED_TEXT + "\n![figure](<images/figure (1).png>)\n",
                "images/figure (1).png": b"image",
            }
        ),
    )
    path, _ = asyncio.run(parser.parse(pdf, tmp_path / "md"))
    markdown = path.read_text("utf-8")
    assert "paper_assets/images/figure%20%281%29.png" in markdown
    paper = Paper(title="Paper")
    paper.fulltext.markdown_path = str(path)
    assert "data:image/png;base64," in embed_paper_images(paper, tmp_path, markdown)
    assert paper_markdown_files(paper, tmp_path)["paper_assets/images/figure (1).png"] == b"image"
    assert parser.cache_is_current(pdf, path)


@pytest.mark.parametrize("mutation", ["converter", "backend", "version", "image", "markdown"])
def test_parser_cache_invalidates_for_backend_and_resource_changes(tmp_path, mutation):
    class AssetExtractor(FakeExtractor):
        async def extract(self, path):
            document = await super().extract(path)
            document.assets = {"images/fig.png": b"image"}
            return document

    extractor = AssetExtractor(SECTIONED_TEXT + "\n![figure](images/fig.png)", markdown_ready=True)
    parser = PdfMarkdownParser(extractor=extractor)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("source"))
    markdown, _ = asyncio.run(parser.parse(pdf, tmp_path / "md"))
    if mutation == "image":
        (markdown.parent / "paper_assets/images/fig.png").rename(tmp_path / "removed.png")
    elif mutation == "markdown":
        markdown.write_text("old broken markdown")
    elif mutation == "backend":
        extractor.name = "different-backend"
    elif mutation == "version":
        extractor.version = "new-version"
    else:
        meta_path = markdown.with_suffix(".parse.json")
        meta = json.loads(meta_path.read_text())
        meta["converter_version"] = "old-version"
        meta_path.write_text(json.dumps(meta))
    asyncio.run(parser.parse(pdf, tmp_path / "md"))
    assert extractor.calls == 2
    assert (markdown.parent / "paper_assets/images/fig.png").exists()


@pytest.mark.parametrize(
    "files",
    [
        {"../full.md": SECTIONED_TEXT},
        {"no-markdown.txt": "missing"},
    ],
)
def test_mineru_rejects_invalid_result_archive(tmp_path, files):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("source"))
    parser = PdfMarkdownParser(
        backend="mineru",
        mineru_api_key="test-token",
        transport=_mineru_transport(files),
    )
    with pytest.raises(FullTextParseError):
        asyncio.run(parser.parse(pdf, tmp_path / "md"))


def test_mineru_rejects_oversized_uncompressed_result(tmp_path, monkeypatch):
    monkeypatch.setattr("paper_agent.adapters.markdown_parser.MAX_RESULT_BYTES", 5)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf("source"))
    parser = PdfMarkdownParser(
        backend="mineru",
        mineru_api_key="test-token",
        transport=_mineru_transport({"full.md": SECTIONED_TEXT}),
    )
    with pytest.raises(FullTextParseError, match="上限"):
        asyncio.run(parser.parse(pdf, tmp_path / "md"))


@pytest.mark.skipif(not HAS_PYPDF, reason="需要 pypdf")
def test_pypdf_extracts_real_embedded_image(tmp_path):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

    writer = PdfWriter()
    page = writer.add_page(
        PdfReader(io.BytesIO(make_pdf("Readable scientific text. " * 8))).pages[0]
    )
    image = DecodedStreamObject()
    image.set_data(bytes([255, 0, 0] * 4))
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(2),
            NameObject("/Height"): NumberObject(2),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    page["/Resources"][NameObject("/XObject")] = DictionaryObject(
        {NameObject("/Im1"): writer._add_object(image)}
    )
    path = tmp_path / "image.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    markdown, _ = asyncio.run(PdfMarkdownParser(backend="pypdf").parse(path, tmp_path / "md"))
    assert "image_assets/images/page-0001-001.png" in markdown.read_text()
    assert (
        (markdown.parent / "image_assets/images/page-0001-001.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    meta = json.loads(markdown.with_suffix(".parse.json").read_text())
    assert "无法可靠识别公式" in meta["warnings"][0]


if __name__ == "__main__":
    unittest.main()
