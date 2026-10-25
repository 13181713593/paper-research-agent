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
    assert len(stages) == 9
    assert len(result.events) == 9
    assert all(event.scope == "stage" for event in result.events)
    assert all(e.output_count is not None for e in stages)
    assert stages[0].details["plan"]["queries"]
    assert all(e.details["papers"] for e in stages[1:7])
    assert stages[7].details["summaries"]
    assert stages[8].details["report"]["markdown"]
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
    assert source_events["arxiv"].message == "provider timed out"
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
        if e.stage == PipelineStage.FULLTEXT_PROCESSED and e.scope == "paper"
    ]
    assert len(child_events) == 3
    assert all(e.status == ProcessingStatus.PARTIAL and e.message for e in child_events)
    assert all(e.details["papers"][0]["abstract"] for e in child_events)
    assert len(fallback.papers) == 1

    excluded = await ResearchPipeline(deps).run(request(require_fulltext=True))
    fulltext = next(
        e
        for e in excluded.progress_events
        if e.stage == PipelineStage.FULLTEXT_PROCESSED and e.scope == "stage"
    )
    assert fulltext.output_count == 0
    assert fulltext.details["papers"] == []
    assert all(
        e.status == ProcessingStatus.SKIPPED
        for e in excluded.progress_events
        if e.stage == PipelineStage.FULLTEXT_PROCESSED and e.scope == "paper"
    )


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
