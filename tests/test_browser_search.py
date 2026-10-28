"""Discovery provenance, isolated browser threads and UI-to-pipeline handoff."""

import asyncio
import base64
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters.browser_search import BrowserPaperSearch, parse_bing_feed
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.browser_search import (
    BrowserSearchHit,
    BrowserSearchResults,
    clean_result_url,
)
from paper_agent.domain.models import Paper

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"


def hit(**kwargs):
    return BrowserSearchHit(
        title="Graph neural network recommendation",
        url="https://arxiv.org/abs/2501.12345",
        engine="Bing",
        query="graph neural network",
        snippet="Search snippet, not an abstract",
        **kwargs,
    )


def test_tracking_links_resolve_to_papers_and_unsafe_schemes_are_rejected():
    target = "https://arxiv.org/abs/2501.12345"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    assert clean_result_url("https://www.bing.com/ck/a?u=a1" + encoded) == target
    assert (
        clean_result_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F2501.12345"
        )
        == target
    )
    assert clean_result_url("javascript:alert(1)") is None
    assert clean_result_url("https://name:password@example.org/") is None


def test_public_bing_feed_and_pdf_versions_preserve_canonical_paper_identity():
    xml = "<rss><channel><item><title>A paper</title><link>https://arxiv.org/abs/2501.12345</link><description>Snippet only</description></item></channel></rss>"
    result = parse_bing_feed(xml, "research topic", 10)[0]
    assert result.snippet == "Snippet only" and result.abstract is None
    pdf = result.model_copy(update={"url": "https://arxiv.org/pdf/2501.12345v2.pdf"})
    assert result.discovery_identity == pdf.discovery_identity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["graph neural network recommendation", "graph neural network recommendation 2025 paper"],
)
async def test_database_supplement_runs_sources_together_and_keeps_true_provenance(
    monkeypatch, query
):
    count = 0
    ready = asyncio.Event()

    class Source:
        def __init__(self, source):
            self.source_name = source

        async def search(self, plan):
            nonlocal count
            if "2025" in query:
                assert plan.queries[0].filters == {
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                }
            count += 1
            if count == 3:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=3)
            return [
                Paper(
                    title="Graph neural network recommendation " + self.source_name,
                    doi="10.1234/" + self.source_name,
                    year=2025,
                )
            ]

    for name, source in [
        ("CrossrefRetriever", "crossref"),
        ("OpenAlexRetriever", "openalex"),
        ("SemanticScholarRetriever", "semantic_scholar"),
    ]:
        monkeypatch.setattr(
            "paper_agent.adapters.browser_search." + name,
            lambda *a, _source=source, **kw: Source(_source),
        )
    hits, warnings = await BrowserPaperSearch()._database_results([query], 10)
    assert len(hits) == 3 and not warnings
    assert {h.metadata_source for h in hits} == {
        "crossref API",
        "openalex API",
        "semantic_scholar API",
    }
    assert all(h.paper_candidate and h.to_paper().year == 2025 for h in hits)


@pytest.mark.asyncio
async def test_doi_results_use_verified_publisher_metadata_instead_of_snippet(monkeypatch):
    monkeypatch.setattr(
        "paper_agent.adapters.browser_search.http_get_json",
        lambda *a, **kw: {
            "message": {
                "DOI": "10.1234/fixture",
                "title": ["Verified title"],
                "published": {"date-parts": [[2025, 3, 2]]},
                "author": [{"given": "Ada", "family": "Example"}],
            }
        },
    )
    result = hit().model_copy(update={"url": "https://doi.org/10.1234/fixture"})
    assert await BrowserPaperSearch()._enrich_doi(result)
    assert result.publication_year == 2025
    assert result.to_paper().authors[0].name == "Ada Example"
    assert result.metadata_source == "Crossref DOI 记录"


def test_search_snippet_is_never_fabricated_into_abstract_or_publication_year():
    result = hit()
    paper = result.to_paper()
    assert paper.abstract is None and paper.year is None
    assert paper.source_records[0].source == "browser"
    assert paper.source_records[0].raw_metadata["search_snippet"] == result.snippet
    assert "未经全文核验" in result.search_context()
    assert paper.arxiv_id == "2501.12345"
    verified = result.model_copy(
        update={"metadata_verified": True, "abstract": "Verified abstract"}
    )
    assert "Verified abstract" in verified.search_context()
    assert "Search snippet, not an abstract" not in verified.search_context()
    not_paper = result.model_copy(update={"url": "https://example.org/blog"})
    assert not not_paper.paper_candidate
    with pytest.raises(ValueError, match="论文详情页"):
        not_paper.to_paper()


def test_browser_worker_uses_its_own_subprocess_capable_loop_across_research_runs():
    main_thread = get_ident()
    observed = []

    class Service(BrowserPaperSearch):
        async def _search(self, queries, engines, limit):
            observed.append((get_ident(), type(asyncio.get_running_loop()).__name__))
            return BrowserSearchResults(hits=[hit()], queries=queries)

    service = Service()
    for _ in range(2):
        result = asyncio.run(service.search(["graph neural network"]))
        assert result.hits
    assert all(thread != main_thread for thread, _ in observed)
    import sys

    if sys.platform == "win32":
        assert all("Proactor" in name for _, name in observed)


@pytest.mark.asyncio
async def test_engines_and_queries_overlap_and_partial_failure_preserves_results(monkeypatch):
    from playwright import async_api

    class Page:
        async def close(self):
            pass

    class Context:
        async def new_page(self):
            return Page()

        async def route(self, *args):
            pass

    class Browser:
        async def new_context(self, **kwargs):
            return Context()

        async def close(self):
            pass

    class Chromium:
        async def launch(self, **kwargs):
            return Browser()

    class Manager:
        async def __aenter__(self):
            return SimpleNamespace(chromium=Chromium())

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(async_api, "async_playwright", Manager)
    ready = asyncio.Event()
    active = 0
    peak = 0

    class Service(BrowserPaperSearch):
        async def _search_page(self, page, engine, query, limit):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 3:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=3)
            active -= 1
            if engine == "DuckDuckGo" and query == "first":
                raise RuntimeError("challenge fixture")
            return [hit()]

        async def _enrich_page(self, page, result):
            pass

    result = await Service(database_fallback=False)._search(
        ["first", "second"], ["Bing", "DuckDuckGo"], 10
    )
    assert peak == 3
    assert len(result.hits) == 1  # Same paper found by multiple queries/engines.
    assert "challenge fixture" in result.warnings[0]


def test_browser_selection_populates_next_research_candidates_and_query_context(
    monkeypatch, finish_ui_research
):
    agent = PaperResearchAgent(ResearchPipeline(build_demo_dependencies()))
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.session_state["browser_search_result"] = BrowserSearchResults(
        hits=[hit(publication_year=2025)]
    )
    app.run()
    app.multiselect(key="browser_hit_selection").set_value([0]).run()
    next(b for b in app.button if b.label == "加入论文候选").click().run()
    assert len(app.session_state["browser_seed_papers"]) == 1
    next(b for b in app.button if b.label == "用于辅助检索词改写").click().run()
    assert "来源：https://arxiv.org" in app.session_state["browser_search_context"][0]
    app.button[0].click().run()
    finish_ui_research(app)
    assert not app.exception
    result = app.session_state["pipeline_result"]
    assert result.request.seed_papers[0].title == "Graph neural network recommendation"
    assert result.request.idea.search_context
    st.cache_resource.clear()
