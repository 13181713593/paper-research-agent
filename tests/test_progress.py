"""Progress must be timely, retain intermediate data, and remain accurate on failures."""

import asyncio
from dataclasses import replace

import pytest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import (
    Paper,
    PaperSource,
    PipelineRequest,
    PipelineStage,
    ProcessingStatus,
    ResearchIdea,
    SearchPlan,
    SearchQuery,
    SourceRecord,
)


@pytest.mark.asyncio
async def test_bilingual_zero_match_is_success_but_real_source_timeout_stays_failed():
    class Planner:
        async def plan(self, idea, limit):
            return SearchPlan(
                intent_summary=idea.text,
                queries=[
                    SearchQuery(source="arxiv", query="scene text detection", language="en"),
                    SearchQuery(source="arxiv", query="场景文本检测与识别", language="zh"),
                    SearchQuery(source="openalex", query="scene text detection", language="en"),
                ],
            )

    class BilingualSource:
        source_name = "arxiv"

        async def search(self, plan):
            return [] if plan.queries[0].language == "zh" else [Paper(title="Scene text detection")]

    class BrokenSource:
        source_name = "openalex"

        async def search(self, plan):
            raise TimeoutError("network timed out")

    deps = replace(
        build_demo_dependencies(),
        query_planner=Planner(),
        retrievers=[BilingualSource(), BrokenSource()],
    )
    result = await ResearchPipeline(deps).run(request())
    sources = {e.source_name: e for e in result.progress_events if e.scope == "source"}
    arxiv = sources["arxiv"]
    assert arxiv.status == ProcessingStatus.SUCCEEDED
    assert arxiv.output_count == 1 and arxiv.message is None
    assert [(q["status"], q["count"]) for q in arxiv.details["query_results"]] == [
        ("succeeded", 1),
        ("succeeded", 0),
    ]
    assert not any("source arxiv failed" in w for w in result.warnings)
    assert sources["openalex"].status == ProcessingStatus.FAILED
    assert sources["openalex"].error_code == "TimeoutError"


def request(**overrides):
    return PipelineRequest(
        idea=ResearchIdea(text="graph neural networks for recommendation"),
        top_k=1,
        pre_rank_pool_size=3,
        **overrides,
    )


@pytest.mark.asyncio
async def test_fast_source_is_visible_before_slow_source_finishes():
    fast_visible = asyncio.Event()
    received = []

    class FastSource:
        source_name = "arxiv"

        async def search(self, plan):
            return [
                Paper(
                    title="Fast paper",
                    abstract="Fast abstract",
                    source_records=[
                        SourceRecord(
                            source=PaperSource.ARXIV,
                            source_id="fast",
                            raw_metadata={"private_fixture": "do-not-export-raw-payload"},
                        )
                    ],
                )
            ]

    class SlowSource:
        source_name = "openalex"

        async def search(self, plan):
            # This can finish only if the event sink receives the fast result while retrieval runs.
            await asyncio.wait_for(fast_visible.wait(), timeout=3)
            return [Paper(title="Slow paper", abstract="Slow abstract")]

    async def sink(event):
        received.append(event)
        if (
            event.scope == "source"
            and event.source_name == "arxiv"
            and event.status == ProcessingStatus.SUCCEEDED
        ):
            assert event.details["papers"][0]["title"] == "Fast paper"
            assert not any(
                e.scope == "stage" and e.stage == PipelineStage.DEDUPED for e in received
            )
            fast_visible.set()

    deps = replace(build_demo_dependencies(), retrievers=[FastSource(), SlowSource()])
    result = await ResearchPipeline(deps).run(request(), event_sink=sink)
    assert result.status != ProcessingStatus.FAILED
    assert fast_visible.is_set()
    assert all(
        "do-not-export-raw-payload" not in event.model_dump_json()
        for event in result.progress_events
    )
    sources = [e for e in result.progress_events if e.scope == "source"]
    assert [e.source_name for e in sources] == ["arxiv", "openalex"]
    assert all(e.details["queries"] for e in sources)


@pytest.mark.asyncio
async def test_source_snapshots_survive_later_mutation_and_all_stages_have_outputs():
    result = await ResearchPipeline(build_demo_dependencies()).run(request())
    stages = [e for e in result.progress_events if e.scope == "stage"]
    # 下载与转换分段各有一条阶段事件；证据确认仍落库但不作为独立页签。
    assert len(stages) == 10
    assert len(result.events) == 10
    assert all(event.scope == "stage" for event in result.events)
    assert all(e.output_count is not None for e in stages)
    assert stages[0].details["plan"]["queries"]
    assert all(e.details["papers"] for e in stages[1:8])
    assert stages[8].details["summaries"]
    assert stages[9].details["report"]["markdown"]
    source = next(e for e in result.progress_events if e.scope == "source")
    original = source.details["papers"][0]
    assert original["scores"]["final"] is None
    assert original["fulltext"]["markdown_path"] is None
    assert original["sources"] == ["arxiv"]
    assert result.papers[0].scores.final is not None
    assert result.total_duration_ms == sum(e.duration_ms or 0 for e in stages)


@pytest.mark.asyncio
async def test_failed_and_empty_sources_have_distinct_results():
    class Broken:
        source_name = "arxiv"

        async def search(self, plan):
            raise TimeoutError("provider timed out")

    class Empty:
        source_name = "openalex"

        async def search(self, plan):
            return []

    deps = replace(build_demo_dependencies(), retrievers=[Broken(), Empty()])
    result = await ResearchPipeline(deps).run(request())
    assert result.status == ProcessingStatus.FAILED
    source_events = {e.source_name: e for e in result.progress_events if e.scope == "source"}
    assert source_events["arxiv"].status == ProcessingStatus.FAILED
    assert source_events["arxiv"].error_code == "TimeoutError"
    # 进度条展示的是“已观察到的状态”，不直接回放异常原文。
    assert "未在等待时间内" in source_events["arxiv"].message
    assert source_events["openalex"].status == ProcessingStatus.SUCCEEDED
    assert source_events["openalex"].output_count == 0
    assert result.progress_events[-1].stage == PipelineStage.RETRIEVED
    assert result.progress_events[-1].status == ProcessingStatus.FAILED


@pytest.mark.asyncio
async def test_retrieved_papers_remain_available_when_later_stage_fails():
    class BrokenRanker:
        async def rank(self, idea, papers, limit):
            raise ValueError("ranking fixture failed")

    deps = replace(build_demo_dependencies(), pre_ranker=BrokenRanker())
    result = await ResearchPipeline(deps).run(request())
    assert result.status == ProcessingStatus.FAILED
    assert not result.papers
    assert sum(e.output_count for e in result.progress_events if e.scope == "source") == 4
    assert next(e for e in result.progress_events if e.stage == PipelineStage.DEDUPED).details[
        "papers"
    ]
    assert result.progress_events[-1].message == "ranking fixture failed"


@pytest.mark.asyncio
async def test_fulltext_fallback_and_exclusions_have_individual_outputs():
    class NoFulltext:
        async def process(self, paper):
            raise PermissionError("no full text")

    deps = replace(build_demo_dependencies(), fulltext_processor=NoFulltext())
    fallback = await ResearchPipeline(deps).run(request())
    child_events = [
        e
        for e in fallback.progress_events
        if e.stage == PipelineStage.MARKDOWN_CONVERTED and e.scope == "paper"
    ]
    assert len(child_events) == 3
    assert all(e.status == ProcessingStatus.PARTIAL and e.message for e in child_events)
    assert all(e.details["papers"][0]["abstract"] for e in child_events)
    download_events = [
        e
        for e in fallback.progress_events
        if e.stage == PipelineStage.DOWNLOADED and e.scope == "paper"
    ]
    assert len(download_events) == 3
    assert all("no full text" in (e.message or "") for e in download_events)
    assert len(fallback.papers) == 1

    excluded = await ResearchPipeline(deps).run(request(require_fulltext=True))
    fulltext = next(
        e
        for e in excluded.progress_events
        if e.stage == PipelineStage.MARKDOWN_CONVERTED and e.scope == "stage"
    )
    assert fulltext.output_count == 0
    assert fulltext.details["papers"] == []
    assert all(
        e.status == ProcessingStatus.SKIPPED
        for e in excluded.progress_events
        if e.stage == PipelineStage.MARKDOWN_CONVERTED and e.scope == "paper"
    )


@pytest.mark.asyncio
async def test_split_fulltext_processor_keeps_per_paper_fallback_and_events():
    """分段接口存在时逐篇记录下载与转换，降级语义保持不变。"""

    class SplitProcessor:
        def __init__(self):
            self.downloads = 0
            self.conversions = 0

        async def download(self, paper):
            self.downloads += 1
            if self.downloads > 1:
                raise RuntimeError("provider down")

        async def convert(self, paper):
            self.conversions += 1
            paper.fulltext.markdown_path = f"data/markdown/{paper.paper_id}.md"
            return paper

    processor = SplitProcessor()
    deps = replace(build_demo_dependencies(), fulltext_processor=processor)
    result = await ResearchPipeline(deps).run(request())

    download_events = [
        e
        for e in result.progress_events
        if e.stage == PipelineStage.DOWNLOADED and e.scope == "paper"
    ]
    convert_events = [
        e
        for e in result.progress_events
        if e.stage == PipelineStage.MARKDOWN_CONVERTED and e.scope == "paper"
    ]
    assert processor.downloads == 3
    assert processor.conversions == 1
    assert [e.status for e in download_events].count(ProcessingStatus.PARTIAL) == 2
    assert [e.status for e in download_events].count(ProcessingStatus.SUCCEEDED) == 1
    assert [e.status for e in convert_events].count(ProcessingStatus.PARTIAL) == 2
    assert [e.status for e in convert_events].count(ProcessingStatus.SUCCEEDED) == 1
    assert all(str(e.paper_id) == e.details["papers"][0]["paper_id"] for e in convert_events)
    assert any("provider down" in w for w in result.warnings)
    assert result.stage == PipelineStage.REPORTED


@pytest.mark.asyncio
async def test_source_failure_marks_retrieval_partial_without_hiding_success():
    class Broken:
        source_name = "broken"

        async def search(self, plan):
            raise TimeoutError("provider timeout")

    deps = build_demo_dependencies()
    deps.retrievers = [deps.retrievers[0], Broken()]
    result = await ResearchPipeline(deps).run(request())
    retrieval = next(
        e
        for e in result.progress_events
        if e.scope == "stage" and e.stage == PipelineStage.RETRIEVED
    )
    assert retrieval.status == ProcessingStatus.PARTIAL
    assert retrieval.output_count == 2
    assert len(retrieval.details["papers"]) == 2
    assert retrieval.details["warnings"]


def test_audit_redacts_signed_queries_and_configured_secrets_without_mutation(monkeypatch):
    from paper_agent.application.progress import redact_audit_value

    monkeypatch.setenv("OPENALEX_API_KEY", "fixture-private-key")
    original = {
        "url": "https://cdn.example.org/paper.pdf?Signature=fixture-signature&Key-Pair-Id=fixture-id&Expires=123&doi=10.1000/a",
        "error": "key fixture-private-key; https://api.example.org/?email=contact%40example.org",
        "html": '<a href="https://cdn.example.org/a.pdf?X-Amz-Signature=another-secret&amp;X-Amz-Credential=credential">PDF</a>',
    }
    cleaned = redact_audit_value(original)
    text = str(cleaned)
    assert all(
        secret not in text
        for secret in (
            "fixture-signature",
            "fixture-id",
            "fixture-private-key",
            "contact%40example.org",
            "another-secret",
            'credential"',
        )
    )
    assert "doi=10.1000/a" in cleaned["url"]
    assert "fixture-signature" in original["url"]


def test_fulltext_retry_uses_complete_models_and_keeps_baseline_unchanged(tmp_path):
    import importlib.util
    from pathlib import Path

    from paper_agent.application.progress import paper_snapshot
    from paper_agent.domain.models import PipelineResult, StageEvent

    target = Path(__file__).resolve().parents[1] / "scripts/run_fulltext_audit.py"
    spec = importlib.util.spec_from_file_location("fulltext_audit", target)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    paper = Paper(title="同池论文", openalex_id="W123", doi="10.1000/a")
    paper.fulltext.parse_error = "旧错误"
    paper.fulltext.attempts = [{"stage": "unpaywall", "status": "failed"}]
    from datetime import UTC, datetime

    event = StageEvent(
        stage=PipelineStage.FULLTEXT_PROCESSED,
        scope="paper",
        paper_id=paper.paper_id,
        status=ProcessingStatus.PARTIAL,
        started_at=datetime.now(UTC),
        details={"title": paper.title, "papers": [paper_snapshot(paper)]},
    )
    baseline = PipelineResult(
        status=ProcessingStatus.PARTIAL,
        stage=PipelineStage.REPORTED,
        request=request(),
        metadata_papers=[paper],
        progress_events=[event],
    )
    pool = audit.retry_pool(baseline)
    assert len(pool) == 1 and pool[0].paper_id == paper.paper_id
    assert pool[0].openalex_id == "W123"
    assert pool[0].fulltext.attempts == [] and pool[0].fulltext.parse_error is None
    assert paper.fulltext.parse_error == "旧错误" and len(paper.fulltext.attempts) == 1

    class Processor:
        async def process(self, candidate):
            raise TimeoutError("模拟网络超时")

    captured = []

    async def record(event):
        captured.append(event)

    result = asyncio.run(audit.retry_fulltext(baseline, Processor(), record))
    assert result.stage == PipelineStage.FULLTEXT_PROCESSED
    assert result.papers == [] and result.summaries == [] and result.report is None
    assert captured[0].error_code == "TimeoutError"
    assert "模拟网络超时" in captured[0].message
    audit.export_result(result, tmp_path, mode="same_pool_fulltext_retry")
    assert (tmp_path / "fulltext_retry_result.json").is_file()
    assert not (tmp_path / "pipeline_result.json").exists()
    assert "error_kind" in (tmp_path / "fulltext_attempts.csv").read_text("utf-8-sig")
