import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from paper_agent.adapters.pdf_downloader import HttpPdfDownloader
from paper_agent.domain.errors import NoOpenAccessCopyError, ProviderUnavailableError
from paper_agent.domain.models import Paper

PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def _paper(url: str = "https://example.org/paper.pdf") -> Paper:
    paper = Paper(title="Downloadable paper")
    paper.fulltext.pdf_url = url
    return paper


class RecordingTransport(httpx.MockTransport):
    """MockTransport that records every requested path."""

    def __init__(self, handler):
        self.requests: list[str] = []
        super().__init__(self._record(handler))

    def _record(self, handler):
        def wrapper(request: httpx.Request) -> httpx.Response:
            self.requests.append(request.url.path)
            return handler(request)

        return wrapper


def _pdf_response() -> httpx.Response:
    return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})


class PdfDownloaderTest(unittest.TestCase):
    def _downloader(self, transport, **overrides) -> HttpPdfDownloader:
        defaults = {
            "timeout_seconds": 5.0,
            "max_bytes": 1_000_000,
            "retry_limit": 2,
            "max_redirects": 3,
            "retry_backoff_seconds": 0.001,
            "transport": transport,
        }
        defaults.update(overrides)
        return HttpPdfDownloader(**defaults)

    def test_download_caches_and_reuses_file(self):
        transport = RecordingTransport(lambda request: _pdf_response())
        downloader = self._downloader(transport)
        with tempfile.TemporaryDirectory() as tmp:
            first = asyncio.run(downloader.download(_paper(), Path(tmp)))
            self.assertTrue(first.exists())
            self.assertEqual(first.read_bytes(), PDF_BYTES)
            meta = json.loads(first.with_suffix(".json").read_text("utf-8"))
            self.assertEqual(meta["bytes"], len(PDF_BYTES))
            self.assertEqual(len(meta["sha256"]), 64)
            second = asyncio.run(downloader.download(_paper(), Path(tmp)))
            self.assertEqual(first, second)
            # Second run must not hit the network again.
            self.assertEqual(len(transport.requests), 1)

    def test_rejects_non_pdf_content_type(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"<html>not a pdf</html>", headers={"content-type": "text/html"}
            )

        downloader = self._downloader(RecordingTransport(handler))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderUnavailableError):
                asyncio.run(downloader.download(_paper(), Path(tmp)))
            self.assertEqual(list(Path(tmp).rglob("*.pdf")), [])

    def test_rejects_body_without_pdf_magic(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"GIF89a not really", headers={"content-type": "application/pdf"}
            )

        downloader = self._downloader(RecordingTransport(handler))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ProviderUnavailableError):
            asyncio.run(downloader.download(_paper(), Path(tmp)))

    def test_rejects_oversized_content_length(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=PDF_BYTES,
                headers={
                    "content-type": "application/pdf",
                    "content-length": "9999999",
                },
            )

        downloader = self._downloader(RecordingTransport(handler), max_bytes=1000)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ProviderUnavailableError):
            asyncio.run(downloader.download(_paper(), Path(tmp)))

    def test_follows_bounded_redirect_chain(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final.pdf"})
            return _pdf_response()

        downloader = self._downloader(RecordingTransport(handler))
        with tempfile.TemporaryDirectory() as tmp:
            path = asyncio.run(downloader.download(_paper("https://example.org/start"), Path(tmp)))
            self.assertTrue(path.exists())

    def test_excessive_redirects_are_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "/loop"})

        downloader = self._downloader(RecordingTransport(handler), max_redirects=1, retry_limit=0)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ProviderUnavailableError):
            asyncio.run(downloader.download(_paper("https://example.org/loop"), Path(tmp)))

    def test_retries_transient_5xx_then_succeeds(self):
        state = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["count"] += 1
            if state["count"] == 1:
                return httpx.Response(503, headers={"content-type": "text/plain"})
            return _pdf_response()

        downloader = self._downloader(RecordingTransport(handler), retry_limit=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = asyncio.run(downloader.download(_paper(), Path(tmp)))
            self.assertTrue(path.exists())
            self.assertEqual(state["count"], 2)

    def test_missing_pdf_raises_no_open_access_copy(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, headers={"content-type": "text/plain"})

        downloader = self._downloader(RecordingTransport(handler), retry_limit=0)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(downloader.download(_paper(), Path(tmp)))

    def test_paper_without_url_is_rejected(self):
        downloader = self._downloader(RecordingTransport(lambda request: _pdf_response()))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(downloader.download(Paper(title="No URL"), Path(tmp)))


if __name__ == "__main__":
    unittest.main()
