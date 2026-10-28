"""第三组独立用例与公平并发实验的回归测试。"""

import asyncio
import os
from pathlib import Path
from unittest import mock

import httpx
import pytest
from test_markdown_parser import SECTIONED_TEXT, make_pdf

from paper_agent.adapters import ProcessingComponents
from paper_agent.adapters.arxiv_metadata import ArxivMetadataEnricher
from paper_agent.adapters.llm_report_writer import LlmReportWriter, comparison_markdown
from paper_agent.adapters.llm_summarizer import LlmSummarizer
from paper_agent.application.processing_benchmark import (
    benchmark_pdf_parsing,
    benchmark_to_markdown,
)
from paper_agent.application.processing_workbench import ProcessingWorkbench, papers_from_arxiv_ids
from paper_agent.domain.models import Paper, ResearchIdea

IDEA = ResearchIdea(text="evidence-aware scientific paper summarization")


@pytest.fixture
def papers(tmp_path):
    items = []
    for index in range(2):
        paper = Paper(title=f"Paper {index}")
        path = tmp_path / f"paper-{index}.pdf"
        path.write_bytes(
            make_pdf(f"Paper {index}. " + "Scientific evidence with detailed methodology. " * 10)
        )
        paper.fulltext.local_pdf_path = str(path)
        items.append(paper)
    return items


class FakeProcessor:
    def __init__(self, root, fail_title=None):
        self.root = root
        self.fail_title = fail_title
        self.active = self.peak = 0

    async def process(self, paper):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            if paper.title == self.fail_title:
                raise OSError("模拟单篇下载失败")
            path = self.root / f"{paper.paper_id}.md"
            path.write_text(SECTIONED_TEXT, encoding="utf-8")
            paper.fulltext.markdown_path = str(path)
            return paper
        finally:
            self.active -= 1


def test_workbench_bounds_concurrency_and_preserves_inputs(tmp_path, papers):
    processor = FakeProcessor(tmp_path)
    summarizer = LlmSummarizer()
    components = ProcessingComponents(processor, summarizer, LlmReportWriter())
    workbench = ProcessingWorkbench(lambda _: components)
    progress = []
    run = asyncio.run(workbench.run(IDEA, papers, concurrency=2, progress=progress.append))
    assert run.result.status.value == "succeeded"
    assert processor.peak == 2
    assert len(run.result.summaries) == 2
    assert all(summary.generated_from_fulltext for summary in run.result.summaries)
    assert [p.paper_id for p in run.result.papers] == [p.paper_id for p in papers]
    assert all(p.fulltext.markdown_path is None for p in papers)
    assert [event.stage.value for event in run.result.events] == [
        "fulltext_processed",
        "summarized",
        "reported",
    ]
    assert all(record.summary_ok for record in run.records)
    assert "跨论文对比矩阵" in run.result.report.markdown
    assert run.elapsed_seconds > 0
    assert progress


@pytest.mark.parametrize("count", [21, 100])
def test_large_batch_keeps_download_and_summary_concurrency_bounded(tmp_path, count):
    items = [Paper(title=f"Paper {index}") for index in range(count)]
    processor = FakeProcessor(tmp_path)

    class TrackedSummarizer(LlmSummarizer):
        active = peak = 0

        async def summarize(self, idea, paper):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.001)
                return await super().summarize(idea, paper)
            finally:
                self.active -= 1

    summarizer = TrackedSummarizer()
    components = ProcessingComponents(processor, summarizer, LlmReportWriter())
    run = asyncio.run(ProcessingWorkbench(lambda _: components).run(IDEA, items, concurrency=4))
    assert processor.peak == summarizer.peak == 4
    assert len(run.result.summaries) == count
    assert run.result.request.top_k == count
    assert [p.paper_id for p in run.result.papers] == [p.paper_id for p in items]


def test_workbench_rejects_batches_over_one_hundred():
    with pytest.raises(ValueError, match="100"):
        asyncio.run(
            ProcessingWorkbench(mock.Mock()).run(
                IDEA, [Paper(title=f"Paper {index}") for index in range(101)]
            )
        )
    identifiers = "\n".join(f"2601.{index:05d}" for index in range(101))
    with pytest.raises(ValueError, match="100"):
        papers_from_arxiv_ids(identifiers)
    assert len(papers_from_arxiv_ids("\n".join(identifiers.splitlines()[:100]))) == 100


def test_single_failure_does_not_discard_other_papers(tmp_path, papers):
    components = ProcessingComponents(
        FakeProcessor(tmp_path, papers[0].title),
        LlmSummarizer(),
        LlmReportWriter(),
    )
    run = asyncio.run(ProcessingWorkbench(lambda _: components).run(IDEA, papers))
    assert run.result.status.value == "partial"
    assert len(run.result.summaries) == 1
    assert run.result.report.paper_ids == [papers[1].paper_id]
    assert run.records[0].errors
    assert not run.records[0].summary_ok


def test_abstract_fallback_is_labeled(tmp_path):
    paper = Paper(
        title="failed", abstract="Only an abstract is available with no experiment details."
    )
    components = ProcessingComponents(
        FakeProcessor(tmp_path, "failed"),
        LlmSummarizer(),
        LlmReportWriter(),
    )
    run = asyncio.run(ProcessingWorkbench(lambda _: components).run(IDEA, [paper]))
    assert run.result.status.value == "partial"
    assert not run.result.summaries[0].generated_from_fulltext
    assert "仅摘要" in run.result.report.markdown


def test_report_failure_preserves_downloadable_summaries(tmp_path, papers):
    writer = mock.Mock()
    writer.write = mock.AsyncMock(side_effect=RuntimeError("模拟报告服务失败"))
    components = ProcessingComponents(FakeProcessor(tmp_path), LlmSummarizer(), writer)
    run = asyncio.run(ProcessingWorkbench(lambda _: components).run(IDEA, papers))
    assert run.result.report is None
    assert len(run.result.summaries) == 2
    assert run.result.status.value == "partial"
    assert run.result.events[-1].status.value == "failed"


def test_all_failures_return_failed_status(tmp_path):
    components = ProcessingComponents(
        FakeProcessor(tmp_path, "failed"), LlmSummarizer(), LlmReportWriter()
    )
    run = asyncio.run(ProcessingWorkbench(lambda _: components).run(IDEA, [Paper(title="failed")]))
    assert run.result.status.value == "failed"
    assert run.result.stage.value == "failed"
    assert run.result.report is None
    assert not run.result.summaries


@pytest.mark.parametrize("workers", [0, 9])
def test_workbench_rejects_invalid_concurrency(workers, papers):
    with pytest.raises(ValueError):
        asyncio.run(ProcessingWorkbench(mock.Mock()).run(IDEA, papers, concurrency=workers))


def test_arxiv_input_is_validated_and_deduplicated():
    items = papers_from_arxiv_ids("1706.03762，1409.0473\n1706.03762")
    assert len(items) == 2
    assert items[0].title == "Attention Is All You Need"
    assert papers_from_arxiv_ids("hep-th/9901001v2")[0].arxiv_id == "hep-th/9901001v2"
    with pytest.raises(ValueError):
        papers_from_arxiv_ids("http://127.0.0.1/private")
    with pytest.raises(ValueError):
        papers_from_arxiv_ids("")


def test_comparison_table_escapes_cells_and_marks_missing_summaries():
    paper = Paper(title="Paper | <test>")
    result = comparison_markdown([paper], [])
    assert "Paper \\| &lt;test&gt;" in result
    assert "未生成总结" in result


def test_benchmark_alternates_order_and_uses_new_cache_dirs(papers, tmp_path):
    calls = []

    async def measure(paths, output, workers):
        assert not output.exists()
        calls.append((output, workers))
        seconds = [8, 4, 6, 10][len(calls) - 1]
        return {
            "workers": workers,
            "seconds": seconds,
            "successes": 2,
            "total": 2,
            "output_sha256": ["same-a", "same-b"],
        }

    with mock.patch("paper_agent.application.processing_benchmark._measure", side_effect=measure):
        result = asyncio.run(benchmark_pdf_parsing(papers, tmp_path, workers=2))
    assert [count for _, count in calls] == [1, 2, 2, 1]
    assert len({path for path, _ in calls}) == 4
    assert result["serial_seconds"] == 9
    assert result["parallel_seconds"] == 5
    assert result["speedup"] == 1.8
    assert result["comparable"]
    assert "不含检索、下载和 LLM" in result["scope"]
    assert "操作系统" in result["cache_policy"]
    assert "1.80×" in benchmark_to_markdown(result)


@pytest.mark.parametrize("count", [21, 100, 101])
def test_benchmark_accepts_expanded_batch_and_checks_upper_limit(tmp_path, count):
    items = []
    for index in range(count):
        path = tmp_path / f"paper-{index}.pdf"
        path.write_bytes(make_pdf(f"Paper {index}. " + "Method details and evidence. " * 10))
        paper = Paper(title=f"Paper {index}")
        paper.fulltext.local_pdf_path = str(path)
        items.append(paper)

    async def measure(paths, output, workers):
        assert len(paths) == count
        return {
            "workers": workers,
            "seconds": 2 if workers == 1 else 1,
            "successes": count,
            "total": count,
            "output_sha256": [str(index) for index in range(count)],
        }

    with mock.patch("paper_agent.application.processing_benchmark._measure", side_effect=measure):
        if count > 100:
            with pytest.raises(ValueError, match="100"):
                asyncio.run(benchmark_pdf_parsing(items, tmp_path, workers=2))
        else:
            report = asyncio.run(benchmark_pdf_parsing(items, tmp_path, workers=2))
            assert report["comparable"] and report["speedup"] == 2
            assert len(report["dataset"]) == count


@pytest.mark.parametrize("reason", ["failure", "mismatch", "input_changed"])
def test_benchmark_suppresses_speedup_for_invalid_comparison(papers, tmp_path, reason):
    count = 0

    async def measure(paths, output, workers):
        nonlocal count
        count += 1
        if reason == "input_changed" and count == 1:
            paths[0].write_bytes(b"changed source")
        hashes = ["a", "b"]
        successes = 2
        if count == 2 and reason == "failure":
            hashes[0], successes = None, 1
        if count == 2 and reason == "mismatch":
            hashes[0] = "different"
        return {
            "workers": workers,
            "seconds": 1,
            "successes": successes,
            "total": 2,
            "output_sha256": hashes,
        }

    with mock.patch("paper_agent.application.processing_benchmark._measure", side_effect=measure):
        result = asyncio.run(benchmark_pdf_parsing(papers, tmp_path, workers=2))
    assert not result["comparable"]
    assert result["speedup"] is None


def test_benchmark_rejects_outside_paths_and_duplicate_content(papers, tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(benchmark_pdf_parsing(papers, tmp_path / "other", workers=2))
    papers[1].fulltext.local_pdf_path = papers[0].fulltext.local_pdf_path
    with pytest.raises(ValueError):
        asyncio.run(benchmark_pdf_parsing(papers, tmp_path, workers=2))


def test_real_multiprocess_pdf_benchmark(papers, tmp_path):
    result = asyncio.run(benchmark_pdf_parsing(papers, tmp_path, workers=2))
    assert result["comparable"]
    assert all(row["successes"] == 2 for row in result["measurements"])
    assert result["speedup"] > 0
    assert all(Path(p.fulltext.local_pdf_path).is_file() for p in papers)


def test_progress_counts_completions_not_input_positions(tmp_path, papers):
    async def exercise():
        downloaded_second = asyncio.Event()
        summarized_second = asyncio.Event()

        class OutOfOrderProcessor(FakeProcessor):
            async def process(self, paper):
                if paper.paper_id == papers[0].paper_id:
                    await downloaded_second.wait()
                result = await super().process(paper)
                if paper.paper_id == papers[1].paper_id:
                    downloaded_second.set()
                return result

        class OutOfOrderSummarizer(LlmSummarizer):
            async def summarize(self, idea, paper):
                if paper.paper_id == papers[0].paper_id:
                    await summarized_second.wait()
                result = await super().summarize(idea, paper)
                if paper.paper_id == papers[1].paper_id:
                    summarized_second.set()
                return result

        components = ProcessingComponents(
            OutOfOrderProcessor(tmp_path), OutOfOrderSummarizer(), LlmReportWriter()
        )
        progress = []
        run = await ProcessingWorkbench(lambda _: components).run(
            IDEA, papers, concurrency=2, progress=progress.append
        )
        return run, progress

    run, progress = asyncio.run(exercise())
    for stage in ("全文处理", "单篇总结"):
        messages = [message for message in progress if message.startswith(f"{stage}：已处理")]
        assert len(messages) == 2
        assert "已处理 1/2" in messages[0] and "第 2 篇" in messages[0]
        assert "已处理 2/2" in messages[1] and "第 1 篇" in messages[1]
        assert "成功 2，失败 0，跳过 0" in messages[1]
    assert [p.paper_id for p in run.result.papers] == [p.paper_id for p in papers]
    assert [s.paper_id for s in run.result.summaries] == [p.paper_id for p in papers]


def test_progress_includes_failed_and_skipped_papers(tmp_path, papers):
    components = ProcessingComponents(
        FakeProcessor(tmp_path, papers[0].title), LlmSummarizer(), LlmReportWriter()
    )
    progress = []
    asyncio.run(
        ProcessingWorkbench(lambda _: components).run(IDEA, papers, progress=progress.append)
    )
    downloads = [line for line in progress if line.startswith("全文处理：已处理")]
    summaries = [line for line in progress if line.startswith("单篇总结：已处理")]
    assert len(downloads) == len(summaries) == 2
    assert "已处理 2/2" in downloads[-1] and "失败 1" in downloads[-1]
    assert "已处理 2/2" in summaries[-1] and "跳过 1" in summaries[-1]
    assert any("第 1 篇" in line and "· 跳过" in line for line in summaries)


def _atom_entry(
    identifier, title="Fetched title", *, affiliation="", published="2017-06-12T00:00:00Z"
):
    return f"""<entry>
      <id>http://arxiv.org/abs/{identifier}</id><title>{title}</title>
      <summary>Official abstract describing methods and experimental findings.</summary>
      <published>{published}</published>
      <author><name>Official Author</name>{affiliation}</author>
      <arxiv:doi>10.1234/example</arxiv:doi>
      <arxiv:journal_ref>Conference reference supplied by arXiv</arxiv:journal_ref>
    </entry>"""


def _atom_feed(*entries):
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">' + "".join(entries) + "</feed>"
    )


def test_arxiv_metadata_matches_ids_not_response_order():
    papers = papers_from_arxiv_ids("1706.03762\n1409.0473")
    before = [p.model_copy(deep=True) for p in papers]
    affiliation = "<arxiv:affiliation>Example University</arxiv:affiliation>"
    feed = _atom_feed(
        _atom_entry("1409.0473v7", "Second title"),
        _atom_entry("1706.03762v7", "First title", affiliation=affiliation),
    )

    def respond(request):
        assert request.url.params["id_list"] == "1706.03762,1409.0473"
        assert request.url.params["max_results"] == "2"
        assert "search_query" not in request.url.params
        assert "authorization" not in request.headers
        return httpx.Response(200, text=feed)

    warnings = asyncio.run(
        ArxivMetadataEnricher(transport=httpx.MockTransport(respond)).enrich(papers)
    )
    assert not warnings
    assert [p.title for p in papers] == ["First title", "Second title"]
    assert [p.paper_id for p in papers] == [p.paper_id for p in before]
    assert papers[0].authors[0].name == "Official Author"
    assert papers[0].institutions == ["Example University"]
    assert not papers[1].institutions  # 数据源未返回机构时不猜测。
    assert papers[0].year == 2017
    assert papers[0].abstract.startswith("Official abstract")
    assert papers[0].is_open_access
    assert papers[0].venue is None  # 不从 arXiv 分类或备注推断正式出版。
    assert str(papers[0].fulltext.pdf_url).endswith("1706.03762v7")
    assert papers[0].source_records[0].source_id == "1706.03762v7"
    assert papers[0].source_records[0].raw_metadata["metadata_method"] == "arxiv_atom_api"


@pytest.mark.parametrize("count", [21, 100])
def test_arxiv_large_batch_is_serial_throttled_and_preserves_order(count):
    papers = papers_from_arxiv_ids("\n".join(f"2601.{index:05d}" for index in range(count)))
    original_ids = [paper.paper_id for paper in papers]
    calls = []
    active = peak = 0

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            identifiers = request.url.params["id_list"].split(",")
            assert len(identifiers) <= 20
            calls.append(identifiers)
            return httpx.Response(
                200,
                text=_atom_feed(
                    *(
                        _atom_entry(identifier + "v1", "Title " + identifier)
                        for identifier in reversed(identifiers)
                    )
                ),
            )
        finally:
            active -= 1

    with mock.patch(
        "paper_agent.adapters.arxiv_metadata.asyncio.sleep", new_callable=mock.AsyncMock
    ) as pause:
        warnings = asyncio.run(
            ArxivMetadataEnricher(transport=httpx.MockTransport(respond)).enrich(papers)
        )
    assert not warnings
    assert peak == 1
    assert [identifier for batch in calls for identifier in batch] == [p.arxiv_id for p in papers]
    assert [p.paper_id for p in papers] == original_ids
    assert all(p.title == "Title " + p.arxiv_id and p.authors and p.abstract for p in papers)
    assert pause.await_count == len(calls) - 1
    assert all(call.args == (3.0,) for call in pause.await_args_list)


def test_arxiv_batch_failure_keeps_successful_batches():
    papers = papers_from_arxiv_ids("\n".join(f"2601.{index:05d}" for index in range(41)))
    originals = [p.model_copy(deep=True) for p in papers]
    calls = []

    def respond(request):
        identifiers = request.url.params["id_list"].split(",")
        calls.append(identifiers)
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(
            200, text=_atom_feed(*(_atom_entry(identifier + "v1") for identifier in identifiers))
        )

    warnings = asyncio.run(
        ArxivMetadataEnricher(
            request_interval_seconds=0, transport=httpx.MockTransport(respond)
        ).enrich(papers)
    )
    assert len(calls) == 3
    assert len(warnings) == 1 and "第 2 批" in warnings[0]
    assert all(p.authors for p in [*papers[:20], papers[40]])
    assert papers[20:40] == originals[20:40]


def test_arxiv_metadata_rejects_more_than_one_hundred_ids():
    papers = [Paper(title="Paper", arxiv_id=f"2601.{index:05d}") for index in range(101)]
    enricher = ArxivMetadataEnricher(transport=httpx.MockTransport(mock.Mock()))
    with pytest.raises(ValueError, match="100"):
        asyncio.run(enricher.enrich(papers))


def test_arxiv_metadata_does_not_mix_versions_or_papers():
    papers = papers_from_arxiv_ids("1706.03762v1\n1409.0473")
    original = papers[0].model_copy(deep=True)
    feed = _atom_feed(_atom_entry("1706.03762v7"), _atom_entry("1409.0473v7", "Right paper"))
    enricher = ArxivMetadataEnricher(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=feed))
    )
    warnings = asyncio.run(enricher.enrich(papers))
    assert len(warnings) == 1
    assert papers[0] == original
    assert papers[1].title == "Right paper"


@pytest.mark.parametrize(
    "body",
    [
        "<broken",
        "<html>not an Atom feed</html>",
        '<!DOCTYPE feed [<!ENTITY e "entity">]><feed>&e;</feed>',
        _atom_feed(_atom_entry("1706.03762v7", published="not-a-date")),
    ],
)
def test_arxiv_bad_payload_keeps_original_metadata(body):
    papers = papers_from_arxiv_ids("1706.03762")
    before = papers[0].model_copy(deep=True)
    enricher = ArxivMetadataEnricher(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    )
    assert asyncio.run(enricher.enrich(papers))
    assert papers[0] == before


def test_arxiv_metadata_timeout_does_not_block_fulltext(tmp_path, papers):
    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    papers[0].arxiv_id = "1706.03762"
    enricher = ArxivMetadataEnricher(transport=httpx.MockTransport(timeout))
    components = ProcessingComponents(FakeProcessor(tmp_path), LlmSummarizer(), LlmReportWriter())
    run = asyncio.run(
        ProcessingWorkbench(lambda _: components, metadata_enricher=enricher.enrich).run(
            IDEA, papers
        )
    )
    assert len(run.result.summaries) == 2
    assert run.result.report is not None
    assert all(record.fulltext_ok for record in run.records)
    assert run.result.status.value == "partial"
    assert "基本信息获取失败" in run.result.warnings[0]


def test_workbench_applies_metadata_to_copies_and_processing_records(tmp_path):
    papers = papers_from_arxiv_ids("1706.03762")
    old_title = papers[0].title
    feed = _atom_feed(_atom_entry("1706.03762v7", "Verified title"))
    enricher = ArxivMetadataEnricher(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=feed))
    )
    components = ProcessingComponents(FakeProcessor(tmp_path), LlmSummarizer(), LlmReportWriter())
    run = asyncio.run(
        ProcessingWorkbench(lambda _: components, metadata_enricher=enricher.enrich).run(
            IDEA, papers
        )
    )
    assert papers[0].title == old_title and not papers[0].authors
    assert run.records[0].title == "Verified title"
    assert run.result.papers[0].authors
    assert "Verified title" in run.result.report.markdown


@pytest.mark.parametrize("status", [404, 429, 503])
def test_arxiv_metadata_http_failure_is_explicit(status):
    papers = papers_from_arxiv_ids("1706.03762")
    enricher = ArxivMetadataEnricher(
        transport=httpx.MockTransport(lambda _: httpx.Response(status))
    )
    assert "获取失败" in asyncio.run(enricher.enrich(papers))[0]
    assert not papers[0].authors


def test_arxiv_metadata_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr("paper_agent.adapters.arxiv_metadata._MAX_RESPONSE_BYTES", 8)
    enricher = ArxivMetadataEnricher(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=_atom_feed(_atom_entry("1706.03762v7")))
        )
    )
    assert asyncio.run(enricher.enrich(papers_from_arxiv_ids("1706.03762")))


@pytest.mark.skipif(os.getenv("RUN_NETWORK_SMOKE") != "1", reason="显式开启才下载真实公开论文")
def test_public_arxiv_workbench_smoke(tmp_path, monkeypatch):
    from paper_agent.bootstrap import build_processing_workbench
    from paper_agent.config import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("MINERU_API_KEY", "")
    settings = Settings(data_dir=str(tmp_path), markdown_backend="pypdf")
    papers = papers_from_arxiv_ids("1706.03762\n1409.0473")
    run = asyncio.run(build_processing_workbench(settings).run(IDEA, papers, concurrency=2))
    assert run.result.status.value == "succeeded", run.result.warnings
    assert len(run.result.summaries) == 2
    assert all(summary.generated_from_fulltext for summary in run.result.summaries)
    assert all(
        paper.authors and paper.abstract and paper.source_records for paper in run.result.papers
    )
    assert "跨论文对比矩阵" in run.result.report.markdown
    print({"真实元数据": [(p.title, len(p.authors), len(p.abstract)) for p in run.result.papers]})
    report = asyncio.run(benchmark_pdf_parsing(run.result.papers, tmp_path, workers=2))
    assert report["comparable"]
    print(
        {
            "全文成功": len(run.result.summaries),
            "处理秒": round(run.elapsed_seconds, 3),
            "串行解析秒": round(report["serial_seconds"], 3),
            "并行解析秒": round(report["parallel_seconds"], 3),
            "加速比": round(report["speedup"], 3),
        }
    )


@pytest.mark.skipif(
    os.getenv("RUN_LLM_SMOKE") != "1",
    reason="仅显式开启时发送公开论文材料并消耗真实 API 额度",
)
def test_live_llm_summary_and_report_without_fallback(monkeypatch, tmp_path):
    from dataclasses import replace

    from paper_agent.adapters.short_review import ShortReviewWriter, review_length
    from paper_agent.bootstrap import build_processing_workbench
    from paper_agent.config import Settings
    from paper_agent.ui.markdown_exports import summary_to_markdown

    assert (os.getenv("OPENAI_API_KEY") or "").strip(), "请通过本地配置注入 API Key"
    monkeypatch.setenv("MINERU_API_KEY", "")
    settings = replace(Settings.from_env(), data_dir=str(tmp_path), markdown_backend="pypdf")
    idea = ResearchIdea(text="分析 Transformer 的注意力机制、实验表现及研究局限，请用中文总结。")
    papers = papers_from_arxiv_ids("1706.03762")
    # 禁止把抽取式或模板回退当成真实 API 测试成功。
    with (
        mock.patch.object(
            LlmSummarizer, "_extractive_summary", side_effect=AssertionError("发生抽取式回退")
        ) as summary_fallback,
        mock.patch.object(
            ShortReviewWriter, "_fallback", side_effect=AssertionError("发生综述模板回退")
        ) as report_fallback,
    ):
        run = asyncio.run(build_processing_workbench(settings).run(idea, papers, concurrency=1))
        summary_fallback.assert_not_called()
        report_fallback.assert_not_called()
    assert run.result.status.value == "succeeded", run.result.warnings
    assert len(run.result.summaries) == 1
    assert run.result.summaries[0].generated_from_fulltext
    assert run.result.summaries[0].evidence
    assert run.result.report is not None
    assert 1000 <= review_length(run.result.report.executive_summary) <= 1500
    assert "模型综合" in run.result.report.markdown
    paper = run.result.papers[0]
    assert paper.authors and paper.abstract and paper.fulltext.local_pdf_path
    assert paper.fulltext.parser_name == "pypdf"
    source = Path(paper.fulltext.markdown_path).read_text("utf-8")
    (tmp_path / "summary.md").write_text(
        summary_to_markdown(run.result.summaries[0], paper, source_text=source), "utf-8"
    )
    (tmp_path / "research_report.md").write_text(run.result.report.markdown, "utf-8")
    (tmp_path / "result.json").write_text(run.result.model_dump_json(indent=2), "utf-8")
    print(
        {
            "模型": settings.llm_model,
            "真实单篇总结": len(run.result.summaries),
            "真实综合报告": 1,
            "综述字数": review_length(run.result.report.executive_summary),
            "解析后端": paper.fulltext.parser_name,
            "发生降级": False,
            "产物目录": str(tmp_path),
        }
    )
