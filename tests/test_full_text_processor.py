import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
import pytest

from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.adapters.open_access_resolver import UnpaywallResolver
from paper_agent.adapters.pdf_downloader import HttpPdfDownloader
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


def _processor(
    tmp: str, resolver=None, downloader=None, parser=None, fallback_parser=None
) -> PdfFullTextProcessor:
    return PdfFullTextProcessor(
        resolver=resolver or FakeResolver(),
        downloader=downloader or FakeDownloader(),
        parser=parser or FakeParser(),
        fallback_parser=fallback_parser,
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
            self.assertTrue(Path(paper.fulltext.local_pdf_path).is_file())
            self.assertIsNotNone(paper.fulltext.sha256)
            self.assertIsNone(paper.fulltext.markdown_path)
            self.assertEqual(paper.fulltext.access_status, AccessStatus.OPEN_ACCESS)
            # 重试应只解析本地已下载的 PDF，不重复请求下载。
            parser.failure = None
            processed = asyncio.run(processor.process(paper))
            self.assertTrue(processed.fulltext.markdown_path)
            self.assertIsNone(processed.fulltext.parse_error)

    def test_remote_parse_failure_falls_back_to_local_parser(self):
        remote = FakeParser(failure=ProviderUnavailableError("mineru down"))
        local = FakeParser(quality=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, parser=remote, fallback_parser=local)
            paper = Paper(title="Fallback paper")
            processed = asyncio.run(processor.process(paper))
            self.assertTrue(processed.fulltext.markdown_path)
            self.assertIsNone(processed.fulltext.parse_error)
            self.assertEqual(remote.calls, 1)
            self.assertEqual(local.calls, 1)
            self.assertEqual(processed.fulltext.parse_quality, 0.5)
            parse_attempts = [a for a in processed.fulltext.attempts if a.get("stage") == "parse"]
            self.assertEqual([a["status"] for a in parse_attempts], ["failed", "succeeded"])
            self.assertIn("回退本地", parse_attempts[-1]["reason"])
            self.assertTrue(
                any("本地解析器" in warning for warning in processed.fulltext.parse_warnings)
            )

    def test_fallback_failure_still_raises_with_both_errors(self):
        remote = FakeParser(failure=ProviderUnavailableError("mineru down"))
        local = FakeParser(failure=FullTextParseError("no text layer"))
        with tempfile.TemporaryDirectory() as tmp:
            processor = _processor(tmp, parser=remote, fallback_parser=local)
            paper = Paper(title="Double failure")
            with self.assertRaises(FullTextParseError):
                asyncio.run(processor.process(paper))
            self.assertIn("ProviderUnavailableError", paper.fulltext.parse_error or "")
            self.assertIn("本地降级失败", paper.fulltext.parse_error or "")
            self.assertIsNone(paper.fulltext.markdown_path)

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


def test_cache_upgrade_reuses_pdf_without_resolving_or_downloading(tmp_path):
    class ChangedParser(FakeParser):
        def cache_is_current(self, pdf, markdown):
            return False

    resolver = FakeResolver(failure=RuntimeError("不应再次定位"))
    downloader = FakeDownloader(failure=RuntimeError("不应再次下载"))
    parser = ChangedParser()
    pdf = tmp_path / "pdfs" / "cached.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4 cached")
    markdown = tmp_path / "old.md"
    markdown.write_text("old broken markdown")
    paper = Paper(title="cache upgrade")
    paper.fulltext.local_pdf_path = str(pdf)
    paper.fulltext.markdown_path = str(markdown)
    processor = _processor(str(tmp_path), resolver, downloader, parser)
    result = asyncio.run(processor.process(paper))
    assert parser.calls == 1
    assert resolver.calls == downloader.calls == 0
    assert result.fulltext.markdown_path != str(markdown)


def test_mineru_processing_budget_covers_remote_polling(tmp_path):
    parser = FakeParser()
    parser.parser_name = "mineru"
    processor = _processor(str(tmp_path), parser=parser)
    assert processor._timeout_seconds == 720


def _http_processor(tmp_path, handler, *, email="student@example.edu", api_key=""):
    transport = httpx.MockTransport(handler)
    return PdfFullTextProcessor(
        resolver=UnpaywallResolver(
            email=email,
            transport=transport,
            discover_locations=True,
            openalex_api_key=api_key,
        ),
        downloader=HttpPdfDownloader(transport=transport, retry_limit=0),
        parser=FakeParser(),
        work_dir=tmp_path,
    )


def _pdf_response():
    return httpx.Response(
        200,
        content=b"%PDF-1.4 mocked paper",
        headers={"Content-Type": "application/pdf"},
    )


def test_arxiv_download_failure_tries_unpaywall_and_stops_on_success(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "arxiv.org":
            return httpx.Response(404)
        if request.url.host == "api.unpaywall.org":
            assert request.url.params["email"] == "student@example.edu"
            return httpx.Response(
                200,
                json={
                    "is_oa": True,
                    "best_oa_location": {"url_for_pdf": "https://repo.example.org/download/42"},
                },
            )
        assert request.url.host == "repo.example.org"
        return _pdf_response()

    paper = Paper(title="跨渠道兜底", arxiv_id="2401.12345", doi="10.1000/fallback")
    result = asyncio.run(_http_processor(tmp_path, handler).process(paper))
    assert calls == ["arxiv.org", "api.unpaywall.org", "repo.example.org"]
    assert result is paper
    assert result.fulltext.markdown_path
    assert str(result.fulltext.pdf_url) == "https://repo.example.org/download/42"


@pytest.mark.parametrize(
    "failure", ["503", "404", "closed", "invalid_json", "invalid_shape", "timeout"]
)
def test_arxiv_and_unpaywall_failures_continue_to_openalex(tmp_path, failure):
    calls = []

    def handler(request):
        host = request.url.host
        calls.append(host)
        if host == "arxiv.org":
            return httpx.Response(403)
        if host == "api.unpaywall.org":
            if failure == "timeout":
                raise httpx.ReadTimeout("模拟元数据超时", request=request)
            if failure.isdigit():
                return httpx.Response(int(failure))
            if failure == "invalid_json":
                return httpx.Response(200, text="not JSON")
            return httpx.Response(200, json=[] if failure == "invalid_shape" else {"is_oa": False})
        if host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "is_oa": True,
                        "pdf_url": "https://open.example.org/paper.pdf",
                    },
                },
            )
        assert host == "open.example.org"
        return _pdf_response()

    paper = Paper(title="OpenAlex 兜底", arxiv_id="2401.12345", doi="10.1000/fallback")
    result = asyncio.run(_http_processor(tmp_path, handler).process(paper))
    assert calls == ["arxiv.org", "api.unpaywall.org", "api.openalex.org", "open.example.org"]
    assert result.fulltext.access_status == AccessStatus.OPEN_ACCESS
    assert result.fulltext.parse_error is None
    assert any(
        a["stage"] == "unpaywall" and a["status"] == "failed" for a in result.fulltext.attempts
    )


def test_unpaywall_pdf_download_failure_continues_to_openalex(tmp_path):
    calls = []

    def handler(request):
        host = request.url.host
        calls.append(host)
        if host == "api.unpaywall.org":
            return httpx.Response(
                200,
                json={
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_pdf": "https://broken.example.org/paper.pdf",
                        "license": "cc-by",
                        "version": "acceptedVersion",
                    },
                },
            )
        if host == "broken.example.org":
            # HTTP 200 也可能只是 HTML，不得作为有效 PDF 接受。
            return httpx.Response(
                200, text="<html>login</html>", headers={"Content-Type": "text/html"}
            )
        if host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "locations": [{"is_oa": True, "pdf_url": "https://open.example.org/paper.pdf"}],
                },
            )
        assert host == "open.example.org"
        return _pdf_response()

    paper = Paper(title="非 arXiv 论文", doi="10.1000/fallback")
    result = asyncio.run(_http_processor(tmp_path, handler).process(paper))
    assert result.fulltext.markdown_path
    assert result.fulltext.license is None
    assert result.fulltext.version is None
    assert calls == [
        "api.unpaywall.org",
        "broken.example.org",
        "api.openalex.org",
        "open.example.org",
    ]


@pytest.mark.parametrize("success_at_fifth", [True, False])
def test_more_than_four_candidates_do_not_block_later_copies_or_channels(
    tmp_path, success_at_fifth
):
    paths = []

    def handler(request):
        paths.append(str(request.url))
        if request.url.host == "broken.example.org":
            if success_at_fifth and request.url.path == "/4.pdf":
                return _pdf_response()
            return httpx.Response(404)
        if request.url.host == "api.unpaywall.org":
            return httpx.Response(404)
        if request.url.host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "pdf_url": "https://open.example.org/paper.pdf",
                        "is_oa": True,
                    },
                },
            )
        assert request.url.host == "open.example.org"
        return _pdf_response()

    paper = Paper(title="第五个候选也应尝试", doi="10.1000/fallback")
    paper.fulltext.candidate_urls = [f"https://broken.example.org/{i}.pdf" for i in range(5)]
    assert asyncio.run(_http_processor(tmp_path, handler).process(paper)).fulltext.markdown_path
    assert paths[:5] == [f"https://broken.example.org/{i}.pdf" for i in range(5)]
    assert len(paths) == (5 if success_at_fifth else 8)
    assert len(paths) == len(set(paths))


def test_multiple_oa_landing_pages_are_tried_without_leaking_openalex_key(tmp_path):
    calls = []
    key = "fixture-not-a-real-openalex-key"

    def handler(request):
        calls.append(request.url.path)
        if request.url.host == "api.openalex.org":
            assert request.headers["Authorization"] == f"Bearer {key}"
            return httpx.Response(
                200,
                json={
                    "locations": [
                        {"is_oa": True, "landing_page_url": "https://repo.example.org/first"},
                        {"is_oa": True, "landing_page_url": "https://repo.example.org/second"},
                        {"is_oa": True, "landing_page_url": "https://repo.example.org/second"},
                    ]
                },
            )
        assert "Authorization" not in request.headers
        if request.url.path == "/first":
            return httpx.Response(503)
        if request.url.path == "/second":
            return httpx.Response(200, text='<meta name="citation_pdf_url" content="/final.pdf">')
        assert request.url.path == "/final.pdf"
        return _pdf_response()

    paper = Paper(title="多个 OA 落地页", doi="10.1000/fallback")
    result = asyncio.run(_http_processor(tmp_path, handler, email="", api_key=key).process(paper))
    assert calls == ["/works/https://doi.org/10.1000/fallback", "/first", "/second", "/final.pdf"]
    assert key not in result.model_dump_json()
    assert any(
        a["stage"] == "unpaywall" and a["status"] == "skipped" for a in result.fulltext.attempts
    )


def test_openalex_failure_still_tries_unpaywall_landing_page(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "api.unpaywall.org":
            return httpx.Response(
                200,
                json={
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_landing_page": "https://repo.example.org/article",
                    },
                },
            )
        if request.url.host == "api.openalex.org":
            return httpx.Response(503)
        if request.url.path == "/article":
            return httpx.Response(
                200, text='<a href="/paper.pdf">PDF</a>', headers={"Content-Type": "text/html"}
            )
        assert request.url.path == "/paper.pdf"
        return _pdf_response()

    result = asyncio.run(
        _http_processor(tmp_path, handler).process(Paper(title="仅落地页", doi="10.1000/a"))
    )
    assert result.fulltext.markdown_path
    assert calls == [
        "api.unpaywall.org",
        "repo.example.org",
        "api.openalex.org",
        "repo.example.org",
        "repo.example.org",
    ]


def test_all_channels_failed_records_error_and_preserves_abstract(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(404)

    paper = Paper(
        title="无公开版本", abstract="原始摘要", doi="10.1000/no-copy", arxiv_id="2401.12345"
    )
    with pytest.raises(NoOpenAccessCopyError):
        asyncio.run(_http_processor(tmp_path, handler).process(paper))
    assert calls == ["arxiv.org", "api.unpaywall.org", "api.openalex.org", "doi.org"]
    assert paper.fulltext.access_status == AccessStatus.UNAVAILABLE
    assert paper.fulltext.parse_error
    assert paper.abstract == "原始摘要"
    assert paper.fulltext.local_pdf_path is None
    assert paper.fulltext.markdown_path is None


def test_legacy_resolver_error_can_still_discover_a_working_copy(tmp_path):
    class LegacyResolver(FakeResolver):
        async def discover(self, paper):
            paper.fulltext.candidate_urls = ["https://example.org/paper.pdf"]

    resolver = LegacyResolver(failure=ProviderUnavailableError("初次定位失败"))
    paper = Paper(title="兼容旧定位器")
    result = asyncio.run(_processor(str(tmp_path), resolver=resolver).process(paper))
    assert result.fulltext.markdown_path
    assert resolver.calls == 1


def test_openalex_id_without_doi_can_locate_pdf(tmp_path):
    def handler(request):
        if request.url.host == "api.openalex.org":
            assert request.url.path == "/works/W123"
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "is_oa": True,
                        "pdf_url": "https://repo.example.org/paper.pdf",
                    },
                },
            )
        assert request.url.host == "repo.example.org"
        return _pdf_response()

    paper = Paper(title="只有 OpenAlex ID", openalex_id="W123")
    assert asyncio.run(_http_processor(tmp_path, handler).process(paper)).fulltext.markdown_path


def test_shared_resolver_keeps_concurrent_paper_fallbacks_independent(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "arxiv.org":
            return httpx.Response(404)
        if request.url.host == "api.unpaywall.org":
            suffix = "first" if request.url.path.endswith("first") else "second"
            return httpx.Response(
                200,
                json={
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_pdf": f"https://repo.example.org/{suffix}.pdf",
                    },
                },
            )
        assert request.url.host == "repo.example.org"
        return _pdf_response()

    processor = _http_processor(tmp_path, handler)
    papers = [
        Paper(title="第一篇", doi="10.1000/first", arxiv_id="2401.12345"),
        Paper(title="第二篇", doi="10.1000/second", arxiv_id="2401.12346"),
    ]

    async def run():
        return await asyncio.gather(*(processor.process(p) for p in papers))

    results = asyncio.run(run())
    assert len(calls) == 6
    assert str(results[0].fulltext.pdf_url) == "https://repo.example.org/first.pdf"
    assert str(results[1].fulltext.pdf_url) == "https://repo.example.org/second.pdf"
    assert all(p.fulltext.markdown_path for p in results)
    assert results[0].fulltext.local_pdf_path != results[1].fulltext.local_pdf_path


def test_timeout_closes_staged_iterator_without_querying_later_channels(tmp_path):
    closed = []
    queried = []

    class StagedResolver(FakeResolver):
        async def iter_candidate_batches(self, paper):
            try:
                yield ["https://example.org/paper.pdf"]
                queried.append("下一渠道")
            finally:
                closed.append(True)

    class SlowDownloader(FakeDownloader):
        async def download(self, paper, destination_dir):
            await asyncio.sleep(1)

    processor = PdfFullTextProcessor(
        resolver=StagedResolver(),
        downloader=SlowDownloader(),
        parser=FakeParser(),
        work_dir=tmp_path,
        timeout_seconds=0.01,
    )
    paper = Paper(title="超时取消")
    with pytest.raises(TimeoutError, match="超过0.01秒预算"):
        asyncio.run(processor.process(paper))
    assert paper.fulltext.attempts[-1]["error_type"] == "TimeoutError"
    assert closed == [True]
    assert queried == []
    assert "超过" in paper.fulltext.parse_error


def test_known_doi_is_scanned_even_when_provider_landing_is_an_aggregator(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "api.unpaywall.org":
            return httpx.Response(200, json={"is_oa": False})
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"locations": []})
        if request.url.host == "aggregator.example.org":
            return httpx.Response(200, text="abstract only")
        if request.url.host == "doi.org":
            return httpx.Response(
                302, headers={"Location": "https://journal.example.org/article/141"}
            )
        if request.url.path == "/article/141":
            return httpx.Response(200, text='<a href="/article/view/141/62"><span>PDF</span></a>')
        if request.url.path == "/article/view/141/62":
            return httpx.Response(
                200,
                text='<iframe src="/files/paper.pdf"></iframe>',
                headers={"Content-Type": "text/html"},
            )
        assert request.url.path == "/files/paper.pdf"
        return _pdf_response()

    paper = Paper(
        title="有出版社 PDF 入口",
        doi="10.71335/example",
        landing_page_url="https://aggregator.example.org/paper/141",
    )
    result = asyncio.run(_http_processor(tmp_path, handler).process(paper))
    assert result.fulltext.markdown_path
    assert str(result.fulltext.pdf_url) == "https://journal.example.org/files/paper.pdf"
    assert "https://doi.org/10.71335/example" in calls
    assert all("citationstyle" not in url for url in calls)


def test_unavailable_channel_is_not_misreported_as_no_lawful_copy(tmp_path):
    def handler(request):
        if request.url.host == "api.unpaywall.org":
            return httpx.Response(404)
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"locations": []})
        raise httpx.ConnectTimeout("模拟 DOI 落地页不可达", request=request)

    paper = Paper(title="定位请求失败", doi="10.1000/unavailable")
    with pytest.raises(ProviderUnavailableError, match="公开全文定位未完成"):
        asyncio.run(_http_processor(tmp_path, handler).process(paper))
    attempt = next(a for a in paper.fulltext.attempts if a["stage"] == "openalex")
    assert attempt["http_status"] == "200"
    assert attempt["candidate_count"] == "0"
    assert "没有公开 PDF" in attempt["reason"]
    assert paper.fulltext.access_status == AccessStatus.UNAVAILABLE


def test_viewer_cycle_does_not_scan_unbounded_pages(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text='<a href="/viewer">PDF</a>')

    paper = Paper(title="循环查看器", landing_page_url="https://repo.example.org/article")
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(_http_processor(tmp_path, handler, email="").process(paper))
    assert calls.count("https://repo.example.org/viewer") == 2  # 一次下载验证、一次页面扫描
    assert len(calls) == 3


def test_phase_split_throttles_parsing_after_all_downloads_finish(tmp_path):
    """下载阶段先整体跑完，解析阶段被单独限流，慢解析不占下载名额。"""
    download_done = []
    active = {"count": 0, "peak": 0}

    class PerPaperDownloader(FakeDownloader):
        async def download(self, paper, destination_dir):
            await asyncio.sleep(0.01)
            pdf_dir = Path(destination_dir)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            path = pdf_dir / f"{paper.paper_id}.pdf"
            path.write_bytes(b"%PDF-1.4 staged")
            download_done.append(paper.title)
            return path

    class ThrottleProbeParser(FakeParser):
        async def parse(self, pdf_path, destination_dir):
            # 所有下载都在任何解析开始前完成
            assert len(download_done) == 6
            active["count"] += 1
            active["peak"] = max(active["peak"], active["count"])
            try:
                await asyncio.sleep(0.01)
                return await super().parse(pdf_path, destination_dir)
            finally:
                active["count"] -= 1

    processor = PdfFullTextProcessor(
        resolver=FakeResolver(),
        downloader=PerPaperDownloader(),
        parser=ThrottleProbeParser(),
        work_dir=tmp_path,
        parse_concurrency=2,
    )
    papers = [Paper(title=f"论文{i}") for i in range(6)]
    results = asyncio.run(processor.process_many(papers, download_concurrency=6))

    assert [error for _paper, error in results] == [None] * 6
    assert [paper.title for paper, _error in results] == [paper.title for paper in papers]
    assert all(paper.fulltext.markdown_path for paper, _error in results)
    assert active["peak"] == 2


def test_locate_timeout_uses_only_the_locate_budget(tmp_path):
    class SlowDownloader(FakeDownloader):
        async def download(self, paper, destination_dir):
            await asyncio.sleep(1)

    processor = PdfFullTextProcessor(
        resolver=FakeResolver(),
        downloader=SlowDownloader(),
        parser=FakeParser(),
        work_dir=tmp_path,
        timeout_seconds=0.05,
        parse_timeout_seconds=30,
    )
    paper = Paper(title="定位超时")
    with pytest.raises(TimeoutError, match="全文定位和下载超过0.05秒预算"):
        asyncio.run(processor.process(paper))
    assert paper.fulltext.attempts[-1]["stage"] == "fulltext"
    assert paper.fulltext.attempts[-1]["error_type"] == "TimeoutError"


def test_parse_timeout_is_reported_as_parse_stage_and_keeps_local_pdf(tmp_path):
    class HangingParser(FakeParser):
        async def parse(self, pdf_path, destination_dir):
            await asyncio.sleep(1)

    processor = PdfFullTextProcessor(
        resolver=FakeResolver(),
        downloader=FakeDownloader(),
        parser=HangingParser(),
        work_dir=tmp_path,
        parse_timeout_seconds=0.05,
    )
    paper = Paper(title="解析超时")
    with pytest.raises(TimeoutError, match="PDF 解析超过0.05秒预算"):
        asyncio.run(processor.process(paper))
    assert paper.fulltext.local_pdf_path
    assert Path(paper.fulltext.local_pdf_path).is_file()
    assert paper.fulltext.attempts[-1]["stage"] == "parse"
    assert paper.fulltext.attempts[-1]["error_kind"] == "provider_unavailable"


def test_download_and_convert_split_records_channel_and_keeps_pdf(tmp_path):
    """分段接口：下载阶段记录取得渠道，转换阶段单独解析。"""

    class LabelledResolver:
        async def iter_candidate_batches_labelled(self, paper):
            yield "openalex", ["https://europepmc.org/api/paper.pdf"]

    class FailingParser(FakeParser):
        async def parse(self, pdf_path, destination_dir):
            raise ProviderUnavailableError("parser offline")

    processor = PdfFullTextProcessor(
        resolver=LabelledResolver(),
        downloader=FakeDownloader(),
        parser=FailingParser(),
        work_dir=tmp_path,
    )
    paper = Paper(title="分段渠道")

    async def run():
        await processor.download(paper)
        assert paper.fulltext.local_pdf_path
        assert paper.fulltext.attempts[-1]["channel"] == "openalex"
        with pytest.raises(ProviderUnavailableError):
            await processor.convert(paper)

    asyncio.run(run())
    assert paper.fulltext.attempts[-1]["stage"] == "parse"
    # 解析失败仍保留下载的 PDF，供重试。
    assert Path(paper.fulltext.local_pdf_path).is_file()


def test_cached_markdown_skips_conversion(tmp_path):
    """已有有效 Markdown 缓存时，转换阶段不应重复解析。"""

    parser = FakeParser()
    processor = PdfFullTextProcessor(
        resolver=FakeResolver(), downloader=FakeDownloader(), parser=parser, work_dir=tmp_path
    )
    paper = Paper(title="缓存命中")
    asyncio.run(processor.process(paper))
    calls = parser.calls

    repaper = Paper(
        title="缓存命中",
        fulltext=paper.fulltext.model_copy(update={"markdown_path": paper.fulltext.markdown_path}),
    )
    asyncio.run(processor.download(repaper))
    asyncio.run(processor.convert(repaper))
    assert parser.calls == calls


if __name__ == "__main__":
    unittest.main()
