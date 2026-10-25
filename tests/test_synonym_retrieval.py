"""Synonyms must reach actual provider parameters, not only the planning UI."""

import pytest

from paper_agent.adapters._concept_extraction import ExtractedConcepts
from paper_agent.adapters.arxiv_retriever import ArxivRetriever, ArxivSearchConfig
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.query_expansion import search_variants
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.domain.models import ResearchIdea


@pytest.mark.asyncio
async def test_late_synonyms_are_used_and_ambiguous_ocr_is_qualified():
    class Extractor:
        def extract(self, text):
            return ExtractedConcepts(
                concepts=["scene text detection", "scene text recognition"],
                synonyms=["STR", "OCR", "text spotting", "E2E text spotting"],
            )

    plan = await HeuristicQueryPlanner(Extractor(), expand_queries=True).plan(
        ResearchIdea(text="场景文本检测与识别"), 30
    )
    assert "text spotting" in plan.query_terms
    assert "end-to-end text spotting" in plan.query_terms
    assert "optical character recognition in natural scenes" in plan.query_terms
    assert "OCR" not in plan.query_terms and "scene text" not in plan.query_terms
    assert len(plan.query_terms) <= 7
    assert plan.source_limit == 30
    for source in {q.source for q in plan.queries}:
        queries = [q for q in plan.queries if q.source == source]
        for term in plan.query_terms:
            assert any(term in q.query for q in queries)
        assert all(q.filters.get("date_from") and q.filters.get("date_to") for q in queries)
        assert len(queries) <= 14


def test_synonym_budget_deduplicates_case_and_explains_omissions():
    terms, notes = search_variants(
        ["graph neural network", "node classification", "graph representation learning"],
        [
            "GNN",
            "Graph Neural Network",
            "graph neural networks",
            "message passing networks",
            "graph convolution networks",
            "graph attention networks",
            "XYZ",
        ],
    )
    assert len(terms) == len(set(terms)) == 6
    assert "message passing networks" in terms
    assert "缩写含义不明确" in notes["XYZ"]
    assert "本轮未采用" in notes["graph attention networks"]
    for note in notes.values():
        if note.startswith("用于检索："):
            assert note.removeprefix("用于检索：") in terms


@pytest.mark.asyncio
async def test_english_topic_does_not_search_component_words_independently():
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="scene text detection"), 30
    )
    assert not {"scene", "text", "detection"} & set(plan.query_terms)
    assert "text spotting" in plan.query_terms


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,adapter,module,param",
    [
        ("crossref", CrossrefRetriever, "crossref_retriever", "query.bibliographic"),
        ("openalex", OpenAlexRetriever, "openalex_retriever", "search"),
        ("europe_pmc", EuropePmcRetriever, "europepmc_retriever", "query"),
        ("semantic_scholar", SemanticScholarRetriever, "semantic_scholar_retriever", "query"),
    ],
)
async def test_every_planned_alias_reaches_http_parameters(
    source, adapter, module, param, monkeypatch
):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["params"])
        return {"message": {"items": []}, "results": [], "data": [], "resultList": {"result": []}}

    monkeypatch.setattr(f"paper_agent.adapters.{module}.http_get_json", get)
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="场景文本检测与识别"), 30
    )
    for query in [q for q in plan.queries if q.source == source]:
        await adapter().search(plan.model_copy(update={"queries": [query]}))
    for term in plan.query_terms:
        assert any(term in params[param] for params in calls)
    if source == "europe_pmc":
        assert all("TITLE_ABS:" in params[param] for params in calls)
        assert all("FIRST_PDATE:" in params[param] for params in calls)


@pytest.mark.asyncio
async def test_arxiv_synonyms_are_separate_queries_with_all_constraints():
    calls = []

    class Client:
        def results(self, search):
            calls.append(search.query)
            return iter([])

    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(
            text="场景文本检测与识别", required_terms=["transformer"], excluded_terms=["medical"]
        ),
        30,
    )
    retriever = ArxivRetriever(client=Client(), config=ArxivSearchConfig(delay_seconds=0))
    for query in [q for q in plan.queries if q.source == "arxiv"]:
        await retriever.search(plan.model_copy(update={"queries": [query]}))
    for term in plan.query_terms:
        expected = f'all:"{term}"' if " " in term else f"all:{term}"
        assert any(expected in expression for expression in calls)
    assert all('AND all:"transformer"' in expression for expression in calls)
    assert all('ANDNOT all:"medical"' in expression for expression in calls)
    assert all("submittedDate:[" in expression for expression in calls)
    assert "transformer" not in plan.query_terms
