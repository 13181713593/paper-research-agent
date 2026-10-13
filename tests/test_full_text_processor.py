import asyncio
import tempfile
import unittest
from pathlib import Path

from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.domain.errors import (
    FullTextParseError,
    NoOpenAccessCopyError,
    ProviderUnavailableError,
)
from paper_agent.domain.models import AccessStatus, Paper


class FakeResolver:
    def __init__(self, pdf_url: str = "https://example.org/paper.pdf", failure=None):
        self.pdf_url = pdf_url
        self.failure = failure
        self.calls = 0

    async def resolve(self, paper: Paper) -> Paper:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        paper.fulltext.pdf_url = self.pdf_url
        paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
        return paper


class FakeDownloader:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    async def download(self, paper: Paper, destination_dir: Path) -> Path:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        pdf_dir = Path(destination_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "fake-paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content for hashing")
        return pdf_path


class FakeParser:
    parser_name = "fake-parser"
    parser_version = "9.9"

    def __init__(self, failure=None, quality: float = 0.75):
        self.failure = failure
        self.quality = quality
        self.calls = 0

    async def parse(self, pdf_path: Path, destination_dir: Path) -> tuple[Path, float]:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        markdown_dir = Path(destination_dir)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_dir / f"{pdf_path.stem}.md"
        markdown_path.write_text("# Title\n\n## Abstract\nSome text.\n", "utf-8")
        return markdown_path, self.quality


def _processor(tmp: str, resolver=None, downloader=None, parser=None) -> PdfFullTextProcessor:
    return PdfFullTextProcessor(
        resolver=resolver or FakeResolver(),
        downloader=downloader or FakeDownloader(),
        parser=parser or FakeParser(),
        work_dir=Path(tmp),
    )


class FullTextProcessorTest(unittest.TestCase):
    def test_successful_processing_updates_fulltext_in_place(self):
        resolver, downloader, parser = FakeResolver(), FakeDownloader(), FakeParser()
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, resolver, downloader, parser)
            paper = Paper(title="Processable paper")
            processed = asyncio.run(processor.process(paper))
            self.assertIs(processed, paper)
            self.assertTrue(processed.fulltext.markdown_path)
            self.assertTrue(Path(processed.fulltext.markdown_path).exists())
            self.assertTrue(processed.fulltext.local_pdf_path)
            self.assertEqual(processed.fulltext.parse_quality, 0.75)
            self.assertEqual(processed.fulltext.parser_name, "fake-parser")
            self.assertEqual(processed.fulltext.parser_version, "9.9")
            self.assertEqual(processed.fulltext.access_status, AccessStatus.OPEN_ACCESS)
            self.assertIsNotNone(processed.fulltext.downloaded_at)
            self.assertEqual(len(processed.fulltext.sha256 or ""), 64)
            self.assertIsNone(processed.fulltext.parse_error)

    def test_existing_markdown_short_circuits(self):
        resolver = FakeResolver(failure=RuntimeError("must not be called"))
        downloader = FakeDownloader(failure=RuntimeError("must not be called"))
        with tempfile.TemporaryDirectory() as tmp:
            markdown = Path(tmp) / "already.md"
            markdown.write_text("# Existing\n", "utf-8")
            processor = _processor(tmp, resolver, downloader)
            paper = Paper(title="Already processed")
            paper.fulltext.markdown_path = str(markdown)
            processed = asyncio.run(processor.process(paper))
            self.assertEqual(processed.fulltext.markdown_path, str(markdown))
            self.assertEqual(resolver.calls, 0)
            self.assertEqual(downloader.calls, 0)

    def test_resolver_failure_propagates(self):
        resolver = FakeResolver(failure=NoOpenAccessCopyError("closed"))
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, resolver)
            with self.assertRaises(NoOpenAccessCopyError):
                asyncio.run(processor.process(Paper(title="No OA copy")))

    def test_download_failure_marks_unavailable_and_raises(self):
        downloader = FakeDownloader(failure=ProviderUnavailableError("boom"))
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, downloader=downloader)
            paper = Paper(title="Broken download")
            with self.assertRaises(ProviderUnavailableError):
                asyncio.run(processor.process(paper))
            # In-place observation survives for the abstract-fallback path.
            self.assertEqual(paper.fulltext.access_status, AccessStatus.UNAVAILABLE)

    def test_parse_failure_records_parse_error_and_raises(self):
        parser = FakeParser(failure=FullTextParseError("no text layer"))
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, parser=parser)
            paper = Paper(title="Unparseable")
            with self.assertRaises(FullTextParseError):
                asyncio.run(processor.process(paper))
            self.assertIn("FullTextParseError", paper.fulltext.parse_error or "")

    def test_concurrent_calls_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp)
            papers = [Paper(title=f"Paper {index}") for index in range(5)]

            async def run_all():
                return await asyncio.gather(*(processor.process(p) for p in papers))

            processed = asyncio.run(run_all())
            self.assertEqual(len(processed), 5)
            for paper in processed:
                self.assertTrue(paper.fulltext.markdown_path)
                self.assertEqual(len(paper.fulltext.sha256 or ""), 64)


if __name__ == "__main__":
    unittest.main()
