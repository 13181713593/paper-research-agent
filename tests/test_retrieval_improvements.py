"""Meeting requirements: bounded search, honest UI metadata and provider recovery."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests

from paper_agent.adapters import _retriever_utils as http
from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.openreview_retriever import OpenReviewRetriever
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.application.progress import paper_snapshot
from paper_agent.domain.models import Author, Paper, PipelineRequest, ResearchIdea, Venue
from paper_agent.ui.progress import query_result_row

TOPICS = [
    "大模型智能体",
    "无人机导航与智能控制",
    (
        "请你帮我调研在5G/6G网络环境下，云边协同架构中的任务卸载与计算资源调度算法，"
        "并重点分析不同架构下的资源分配策略"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topic,expected", list(zip(TOPICS, ["language model", "navigation", "offloading"]))
)
async def test_requested_topics_generate_specific_bounded_english_queries(topic, expected):
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(ResearchIdea(text=topic), 30)
    english = [q.query.lower() for q in plan.queries if q.language == "en"]
    assert english and any(expected in q for q in english)
    assert len(plan.queries) <= 48
    assert all(len(q.query) <= 180 for q in plan.queries)
    if len(topic) > 40:
        assert not any(topic in q.query for q in plan.queries)
        assert any("offloading" in q and "edge" in q for q in english)
        assert any("resource" in q and "edge" in q for q in english)
        assert any("5g" in q or "6g" in q for q in english)


def test_errors_are_friendly_without_inventing_cause():
    row = query_result_row(
        {
            "query": "test",
            "status": "failed",
            "count": 0,
            "error": "ProviderRateLimitedError: https://api.example.com 429",
        }
    )
    assert "429" not in row["说明"] and "https://" not in row["说明"]
    assert "频率" in row["说明"]


def test_stage_snapshot_preserves_affiliations_venue_zero_citations_and_abstract():
    p = Paper(
        title="A paper",
        authors=[Author(name="A", affiliations=["Example University"])],
        venue=Venue(name="Example Conference"),
        abstract="An abstract.",
        citation_count=0,
    )
    snapshot = paper_snapshot(p)
    assert snapshot["institutions"] == ["Example University"]
    assert snapshot["citation_count"] == 0
    assert snapshot["venue"] == "Example Conference"
    assert snapshot["abstract"] == "An abstract."


def test_semantic_scholar_maps_affiliations_and_publication_venue():
    p = SemanticScholarRetriever()._to_paper(
        {
            "title": "A paper",
            "authors": [{"name": "A", "affiliations": ["University"]}],
            "publicationVenue": {"name": "Example Proceedings", "type": "conference"},
            "citationCount": 0,
            "abstract": "Abstract",
        }
    )
    assert p.institutions == ["University"]
    assert p.authors[0].affiliations == ["University"]
    assert p.venue.name == "Example Proceedings"


def test_long_retry_after_is_not_shortened_and_does_not_block(monkeypatch):
    get = Mock(return_value=Mock(status_code=429, headers={"Retry-After": "600"}))
    sleep = Mock()
    monkeypatch.setattr(http.requests, "get", get)
    monkeypatch.setattr(http.time, "sleep", sleep)
    with pytest.raises(http.ProviderRateLimitedError):
        http.http_get_json(
            "https://slow-example.invalid/works", max_retries=2, max_retry_after=2, total_timeout=5
        )
    assert get.call_count == 1
    sleep.assert_not_called()
    # A subsequent query must honor the service cooldown as well.
    with pytest.raises(http.ProviderRateLimitedError):
        http.http_get_json("https://slow-example.invalid/works", total_timeout=5)
    assert get.call_count == 1


def test_transient_timeout_recovers_but_auth_failure_does_not_retry(monkeypatch):
    ok = Mock(status_code=200, headers={})
    ok.json.return_value = {"data": []}
    get = Mock(side_effect=[requests.Timeout(), ok])
    monkeypatch.setattr(http.requests, "get", get)
    assert http.http_get_json("https://recover-example.invalid", initial_backoff=0) == {"data": []}
    assert get.call_count == 2
    get.reset_mock(side_effect=True)
    get.return_value = Mock(status_code=401, headers={}, text="private details")
    with pytest.raises(http.ProviderResponseError):
        http.http_get_json("https://auth-example.invalid")
    assert get.call_count == 1


@pytest.mark.asyncio
async def test_source_deadline_preserves_successful_papers_and_stops_later_queries():
    deps = build_demo_dependencies()

    class SlowSource:
        source_name = "openalex"
        calls = 0

        async def search(self, plan):
            self.calls += 1
            if self.calls == 1:
                return [Paper(title="Retained result")]
            await asyncio.sleep(5)
            return []

    slow = SlowSource()
    deps = replace(
        deps,
        query_planner=HeuristicQueryPlanner(expand_queries=True),
        retrievers=[deps.retrievers[0], slow],
    )
    result = await ResearchPipeline(deps, source_timeout_seconds=0.05).run(
        PipelineRequest(idea=ResearchIdea(text=TOPICS[0]), top_k=1, pre_rank_pool_size=2)
    )
    event = next(e for e in result.progress_events if e.source_name == "openalex")
    assert event.status.value == "partial"
    assert event.output_count == 1
    assert slow.calls == 2
    assert any(q["status"] == "skipped" for q in event.details["query_results"])
    assert "TimeoutError" not in event.message


@pytest.mark.asyncio
async def test_exhausted_rate_limit_does_not_retry_every_query():
    deps = build_demo_dependencies()

    class Limited:
        source_name = "openalex"
        calls = 0

        async def search(self, plan):
            self.calls += 1
            raise http.ProviderRateLimitedError("internal diagnostics")

    limited = Limited()
    deps = replace(
        deps,
        query_planner=HeuristicQueryPlanner(expand_queries=True),
        retrievers=[deps.retrievers[0], limited],
    )
    result = await ResearchPipeline(deps).run(
        PipelineRequest(idea=ResearchIdea(text=TOPICS[0]), top_k=1, pre_rank_pool_size=2)
    )
    assert limited.calls == 1
    assert result.papers
    assert all("internal diagnostics" not in w for w in result.warnings)


def test_sorted_search_uses_only_supported_fields(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "paper_agent.adapters.semantic_scholar_retriever.http_get_json",
        lambda url, **kwargs: calls.append(kwargs) or {"data": []},
    )
    SemanticScholarRetriever()._fetch(
        '"task offloading" "edge computing" 5g', 3, sort_by="publication_date"
    )
    assert "authors.affiliations" not in calls[0]["params"]["fields"]
    assert calls[0]["params"]["query"] == '"task offloading" "edge computing" 5g'


def test_europe_pmc_core_metadata_does_not_discard_structured_fields():
    paper = EuropePmcRetriever()._to_paper(
        {
            "title": "Example",
            "authorList": {
                "author": [
                    {
                        "fullName": "Jane Doe",
                        "authorAffiliationDetailsList": {
                            "authorAffiliation": [{"affiliation": "Example University"}]
                        },
                    }
                ]
            },
            "journalInfo": {"journal": {"title": "Journal Example"}, "volume": "12", "issue": "3"},
            "abstractText": "<p>Study &amp; results.</p>",
            "pmid": "1234",
            "citedByCount": "0",
        }
    )
    assert paper.institutions == ["Example University"]
    assert paper.authors[0].affiliations == ["Example University"]
    assert paper.venue.name == "Journal Example"
    assert paper.venue.volume == "12"
    assert paper.abstract == "Study & results."
    assert str(paper.landing_page_url) == "https://europepmc.org/article/MED/1234"


@pytest.mark.parametrize("hint", ["nan", "inf", "-1", "not a date"])
def test_invalid_retry_after_uses_finite_fallback(hint):
    assert http._retry_after(Mock(headers={"Retry-After": hint}), fallback=2.0) == 2.0


def test_service_unavailable_cooldown_is_not_reported_as_rate_limiting(monkeypatch):
    get = Mock(return_value=Mock(status_code=503, headers={"Retry-After": "600"}))
    monkeypatch.setattr(http.requests, "get", get)
    for _ in range(2):
        with pytest.raises(http.ProviderUnavailableError):
            http.http_get_json("https://busy-example.invalid", total_timeout=2)
    assert get.call_count == 1


def test_english_long_prose_does_not_turn_into_an_exact_sentence_query():
    topic = (
        "Please investigate task offloading and resource allocation across edge computing "
        "and cloud computing architectures in 5G and 6G networks and compare their strategies"
    )
    plan = asyncio.run(
        HeuristicQueryPlanner(expand_queries=True).plan(ResearchIdea(text=topic), 30)
    )
    assert all(len(q.query.split()) <= 12 for q in plan.queries)
    assert not any("please" in q.query.lower() for q in plan.queries)
    assert any("offloading" in q.query and "edge" in q.query for q in plan.queries)


@pytest.mark.asyncio
async def test_openreview_is_in_the_plan_and_maps_note_metadata(monkeypatch):
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="大模型智能体"), 10
    )
    queries = [q for q in plan.queries if q.source.value == "openreview"]
    assert queries and all(q.filters.get("date_from") for q in queries)
    payload = {
        "notes": [{
            "id": "abc123", "cdate": 1760000000000,
            "content": {
                "title": {"value": "Large Language Model Agents"},
                "authors": {"value": ["Jane Doe", "John Roe"]},
                "authorids": {"value": ["~Jane_Doe1", "~John_Roe1"]},
                "affiliations": {"value": ["Example University"]},
                "abstract": {"value": "Agent abstract."},
                "venue": {"value": "ICLR 2026"},
                "keywords": {"value": ["agents"]},
                "doi": {"value": "10.1234/example"},
            },
        }]
    }
    monkeypatch.setattr(
        "paper_agent.adapters.openreview_retriever.http_get_json",
        lambda url, **kwargs: payload,
    )
    papers = await OpenReviewRetriever().search(plan.model_copy(update={"queries": [queries[0]]}))
    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Large Language Model Agents"
    assert paper.institutions == ["Example University"]
    assert paper.authors[0].affiliations == ["Example University"]
    assert paper.venue.name == "ICLR 2026"
    assert paper.abstract == "Agent abstract."
    assert paper.doi == "10.1234/example"
    assert str(paper.landing_page_url).endswith("abc123")


def test_openreview_query_uses_supported_params_and_caps_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "paper_agent.adapters.openreview_retriever.http_get_json",
        lambda url, **kwargs: calls.append(kwargs) or {"notes": []},
    )
    retriever = OpenReviewRetriever()
    retriever._fetch(
        '"edge computing"',
        500,
        {"date_from": "2024-01-01", "date_to": "2024-12-31"},
    )
    params = calls[0]["params"]
    assert params["term"] == "edge computing"
    assert params["limit"] == 100
    assert set(params) == {"term", "content", "group", "source", "limit"}


@pytest.mark.asyncio
async def test_openreview_filters_records_outside_requested_date_window(monkeypatch):
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="large language model agents"), 10
    )
    query = next(q for q in plan.queries if q.source.value == "openreview")
    query = query.model_copy(update={"filters": {"date_from": "2025-01-01", "date_to": "2025-12-31"}})
    payload = {
        "notes": [
            {"id": "outside", "cdate": 1704067200000, "content": {"title": {"value": "Old"}}},
            {"id": "inside", "cdate": 1764547200000, "content": {"title": {"value": "New"}}},
        ]
    }
    monkeypatch.setattr(
        "paper_agent.adapters.openreview_retriever.http_get_json",
        lambda url, **kwargs: payload,
    )
    papers = await OpenReviewRetriever().search(plan.model_copy(update={"queries": [query]}))
    assert [p.title for p in papers] == ["New"]
