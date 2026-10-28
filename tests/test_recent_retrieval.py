"""Source-level publication filters, sort parameters and relevance-safe recent quotas."""

import math
from datetime import UTC, date, datetime

import arxiv
import pytest

from paper_agent.adapters.academic_ranker import AcademicFinalRanker, AcademicPreRanker
from paper_agent.adapters.arxiv_retriever import ArxivRetriever, ArxivSearchConfig
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.domain.models import Paper, PaperSource, ResearchIdea, SearchPlan, SearchQuery
from paper_agent.domain.publication_policy import is_recent, reserve_recent, years_before


@pytest.mark.asyncio
async def test_default_plan_uses_rolling_five_years_and_disjoint_recent_cohorts(monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 25, tzinfo=UTC)

    monkeypatch.setattr("paper_agent.adapters.query_planner.datetime", Clock)
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="场景文本检测与识别"), 30
    )
    assert plan.date_from == date(2021, 10, 25)
    assert plan.date_to == date(2026, 10, 25)
    assert plan.recent_from == date(2024, 10, 25)
    for source in PaperSource:
        queries = [q for q in plan.queries if q.source == source]
        if not queries:
            continue
        assert {q.language for q in queries} == {"en", "zh"}
        assert all(
            q.sort_by == ("relevance" if source == PaperSource.CROSSREF else "publication_date")
            for q in queries
        )
        recent = [q for q in queries if q.time_window == "近2年优先"]
        older = [q for q in queries if q.time_window == "近3–5年补充"]
        assert all(q.filters["date_from"] == "2024-10-25" for q in recent)
        assert all(q.filters["date_to"] == "2024-10-24" for q in older)
        assert sum(q.limit for q in recent if q.language == "en") >= math.ceil(30 * 0.7)


@pytest.mark.asyncio
async def test_explicit_historical_date_range_is_preserved():
    idea = ResearchIdea(
        text="scene text detection", date_from=date(2019, 1, 1), date_to=date(2020, 1, 1)
    )
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(idea, 10)
    assert all(
        q.filters["date_from"] == "2019-01-01" and q.filters["date_to"] == "2020-01-01"
        for q in plan.queries
    )
    assert plan.date_from == idea.date_from and plan.date_to == idea.date_to


def test_leap_year_and_uncertain_boundary_year_do_not_invent_dates():
    assert years_before(date(2024, 2, 29), 5) == date(2019, 2, 28)
    assert not is_recent(Paper(title="Year only", year=2024), date(2024, 10, 25))
    assert is_recent(Paper(title="Next year", year=2025), date(2024, 10, 25))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,adapter,module",
    [
        ("crossref", CrossrefRetriever, "crossref_retriever"),
        ("openalex", OpenAlexRetriever, "openalex_retriever"),
        ("europe_pmc", EuropePmcRetriever, "europepmc_retriever"),
        ("semantic_scholar", SemanticScholarRetriever, "semantic_scholar_retriever"),
    ],
)
async def test_dates_and_sort_reach_each_real_http_adapter(source, adapter, module, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return {"message": {"items": []}, "results": [], "data": [], "resultList": {"result": []}}

    monkeypatch.setattr(f"paper_agent.adapters.{module}.http_get_json", get)
    query = SearchQuery(
        source=source,
        query="scene text detection",
        limit=3,
        filters={"date_from": "2024-10-25", "date_to": "2026-10-25"},
        sort_by="publication_date",
    )
    assert await adapter().search(SearchPlan(intent_summary="test", queries=[query])) == []
    url, params = calls[0]
    if source == "crossref":
        assert params["filter"] == "from-pub-date:2024-10-25,until-pub-date:2026-10-25"
        assert params["sort"] == "published" and params["order"] == "desc"
        assert params["query.title"] == query.query
    elif source == "openalex":
        assert params["filter"] == "from_publication_date:2024-10-25,to_publication_date:2026-10-25"
        assert params["sort"] == "publication_date:desc"
    elif source == "europe_pmc":
        assert "FIRST_PDATE:[2024-10-25 TO 2026-10-25]" in params["query"]
        assert params["query"].endswith("sort_date:y")
    else:
        assert url.endswith("/paper/search")
        assert params["publicationDateOrYear"] == "2024-10-25:2026-10-25"
        assert params["sort"] == "publicationDate:desc"
        assert params["limit"] == 3


@pytest.mark.asyncio
async def test_arxiv_uses_first_submission_date_filter_and_newest_first():
    searches = []

    class Client:
        def results(self, search):
            searches.append(search)
            return iter([])

    retriever = ArxivRetriever(client=Client(), config=ArxivSearchConfig(delay_seconds=0))
    query = SearchQuery(
        source="arxiv",
        query='"scene text detection"',
        sort_by="publication_date",
        filters={"date_from": "2024-10-25", "date_to": "2026-10-25"},
    )
    assert await retriever.search(SearchPlan(intent_summary="test", queries=[query])) == []
    search = searches[0]
    assert "submittedDate:[202410250000 TO 202610252359]" in search.query
    assert search.sort_by == arxiv.SortCriterion.SubmittedDate
    assert search.sort_order == arxiv.SortOrder.Descending


@pytest.mark.asyncio
@pytest.mark.parametrize("ranker", [AcademicPreRanker, AcademicFinalRanker])
async def test_recent_quota_is_applied_after_topic_and_five_year_screening(ranker):
    idea = ResearchIdea(
        text="scene text detection",
        date_from=date(2021, 10, 25),
        date_to=date(2026, 10, 25),
        recent_from=date(2024, 10, 25),
    )
    fresh = [
        Paper(title=f"Scene text detection recent {i}", publication_date=date(2025, 1, 1))
        for i in range(5)
    ]
    older = [
        Paper(
            title=f"Scene text detection study {i}",
            publication_date=date(2022, 1, 1),
            citation_count=10000,
        )
        for i in range(5)
    ]
    noise = [
        Paper(title="Cancer detection", publication_date=date(2025, 1, 1)),
        Paper(title="Scene text detection classic", publication_date=date(2017, 1, 1)),
        Paper(title="Scene text detection future", publication_date=date(2027, 1, 1)),
    ]
    result = await ranker().rank(idea, older + fresh + noise, 5)
    assert len(result) == 5
    assert sum(is_recent(p, idea.recent_from) for p in result) >= 4
    assert not {p.paper_id for p in noise} & {p.paper_id for p in result}
    # Lack of related fresh papers must not admit unrelated recent papers.
    limited = await ranker().rank(idea, older + noise, 5)
    assert {p.paper_id for p in limited} == {p.paper_id for p in older}


def test_recent_quota_can_fill_with_older_eligible_candidates_when_recent_are_few():
    older = [Paper(title=f"Old {i}", year=2022) for i in range(5)]
    fresh = [Paper(title="New", year=2025)]
    result = reserve_recent(older + fresh, 5, date(2024, 10, 25))
    assert len(result) == 5 and fresh[0] in result
