"""Prove real blocking I/O overlaps, rate-limit queues stop and ordering stays stable."""

import asyncio
from dataclasses import replace
from threading import Barrier, Lock

import pytest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters._retriever_utils import ProviderRateLimitedError
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import (
    Paper,
    PipelineRequest,
    ProcessingStatus,
    ResearchIdea,
    SearchPlan,
    SearchQuery,
)


def request(**kwargs):
    return PipelineRequest(
        idea=ResearchIdea(text="graph neural network recommendation"),
        top_k=1,
        pre_rank_pool_size=3,
        retrieval_concurrency=4,
        **kwargs,
    )


class TwoSourcePlanner:
    async def plan(self, idea, limit):
        return SearchPlan(
            intent_summary=idea.text,
            queries=[
                SearchQuery(source=source, query=f"query {i}")
                for source in ("crossref", "openalex")
                for i in range(2)
            ],
        )


def test_blocking_requests_for_multiple_queries_and_sources_really_overlap():
    barrier = Barrier(4, timeout=5)
    lock = Lock()
    active = 0
    peak = 0

    class Source:
        def __init__(self, name):
            self.source_name = name

        async def search(self, plan):
            return await asyncio.to_thread(self.blocking, plan.queries[0].query)

        def blocking(self, query):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                return [Paper(title=f"{self.source_name} {query}")]
            finally:
                with lock:
                    active -= 1

    deps = replace(
        build_demo_dependencies(),
        query_planner=TwoSourcePlanner(),
        retrievers=[Source("crossref"), Source("openalex")],
    )
    pipeline = ResearchPipeline(deps, retrieval_concurrency=4, query_concurrency=2)
    # Reuse the cached pipeline across independent event loops, as Streamlit does.
    for _ in range(2):
        result = asyncio.run(pipeline.run(request()))
        assert result.status != ProcessingStatus.FAILED
        sources = [e for e in result.progress_events if e.scope == "source"]
        assert all(e.details["concurrency"]["peak"] == 2 for e in sources)
        assert all(
            [q["query"] for q in e.details["query_results"]] == ["query 0", "query 1"]
            for e in sources
        )
    assert peak == 4


@pytest.mark.asyncio
async def test_persistent_rate_limit_stops_unsent_queries_but_other_sources_finish():
    class Planner:
        async def plan(self, idea, limit):
            return SearchPlan(
                intent_summary=idea.text,
                queries=[
                    *(SearchQuery(source="semantic_scholar", query=f"topic {i}") for i in range(6)),
                    SearchQuery(source="openalex", query="graph neural network"),
                ],
            )

    class Limited:
        source_name = "semantic_scholar"
        calls = 0

        async def search(self, plan):
            self.calls += 1
            raise ProviderRateLimitedError("429 fixture")

    class Working:
        source_name = "openalex"

        async def search(self, plan):
            return [Paper(title="Graph neural network recommendation")]

    source = Limited()
    deps = replace(
        build_demo_dependencies(), query_planner=Planner(), retrievers=[source, Working()]
    )
    result = await ResearchPipeline(deps).run(request())
    assert source.calls == 1
    event = next(e for e in result.progress_events if e.source_name == source.source_name)
    assert [q["status"] for q in event.details["query_results"]] == ["failed"] + ["skipped"] * 5
    assert result.metadata_papers


@pytest.mark.asyncio
async def test_completed_request_order_does_not_change_merge_precedence():
    second_done = asyncio.Event()

    class Source:
        source_name = "crossref"

        async def search(self, plan):
            if plan.queries[0].query.endswith("0"):
                await asyncio.wait_for(second_done.wait(), timeout=3)
                return [Paper(title="Preferred original title", doi="10.1234/same")]
            second_done.set()
            return [Paper(title="Other title", doi="10.1234/same")]

    class Other:
        source_name = "openalex"

        async def search(self, plan):
            return []

    deps = replace(
        build_demo_dependencies(), query_planner=TwoSourcePlanner(), retrievers=[Source(), Other()]
    )
    result = await ResearchPipeline(deps).run(request())
    source = next(e for e in result.progress_events if e.source_name == "crossref")
    assert source.details["papers"][0]["title"] == "Preferred original title"


@pytest.mark.asyncio
async def test_browser_candidates_survive_even_when_api_search_returns_empty():
    class Empty:
        def __init__(self, name):
            self.source_name = name

        async def search(self, plan):
            return []

    deps = replace(build_demo_dependencies(), retrievers=[Empty("crossref"), Empty("openalex")])
    result = await ResearchPipeline(deps).run(
        request(seed_papers=[Paper(title="Browser paper", year=2025)])
    )
    assert any(p.title == "Browser paper" for p in result.metadata_papers)
    assert any(e.source_name == "browser" and e.output_count == 1 for e in result.progress_events)
