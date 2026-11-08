"""Real failure regressions: public PDFs, missing abstracts and report evidence."""

import json
import ssl
from unittest.mock import Mock

import httpx
import pytest

from paper_agent.adapters import _retriever_utils as http
from paper_agent.adapters.llm_concept_extractor import LLMConceptExtractor
from paper_agent.adapters.llm_report_writer import publication_note
from paper_agent.adapters.llm_summarizer import LlmSummarizer
from paper_agent.adapters.open_access_resolver import UnpaywallResolver, _PdfLinks
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.adapters.short_review import ShortReviewWriter
from paper_agent.adapters.source_evidence import (
    extract_abstract,
    has_substantive_description,
    quote_in_source,
)
from paper_agent.adapters.tls import verified_context
from paper_agent.domain.models import Paper, ResearchIdea

IDEA = ResearchIdea(text="5G/6G 云边协同任务卸载与资源分配")
ABSTRACT = (
    "We propose a Q-learning algorithm for task offloading in edge-cloud computing. "
    "The algorithm maximizes the number of tasks meeting their deadlines."
)


@pytest.mark.parametrize("doi", ["10.48550/arXiv.2603.16881", "10.21203/rs.3.rs-7339827/v1"])
def test_preprint_doi_is_labeled_without_claiming_peer_review(doi):
    note = publication_note(Paper(title="Paper", doi=doi))
    assert "预印本" in note and "未核验" in note


@pytest.mark.asyncio
async def test_scitepress_public_pdf_is_found_without_executing_javascript():
    paper = Paper(
        title="Task Offloading in Edge-Cloud Computing Using a Q-Learning Algorithm",
        year=2024,
        doi="10.5220/0012590800003711",
    )
    resolver = UnpaywallResolver(email="")
    await resolver.resolve(paper)
    assert str(paper.fulltext.pdf_url) == "https://www.scitepress.org/Papers/2024/125908/125908.pdf"
    assert paper.fulltext.local_pdf_path is None  # A candidate is not a downloaded PDF.


def test_publisher_hint_does_not_create_candidates_for_other_dois_or_explicitly_closed_work():
    resolver = UnpaywallResolver(email="")
    for doi, closed in [("10.1000/0012590800003711", False), ("10.5220/0012590800003711", True)]:
        paper = Paper(
            title="Other work", doi=doi, year=2024, is_open_access=False if closed else None
        )
        assert resolver._direct_url(paper) is None


@pytest.mark.asyncio
async def test_missing_abstract_is_enriched_from_public_landing_page_and_never_overwritten():
    def handler(request):
        return httpx.Response(200, text=f'<meta name="citation_abstract" content="{ABSTRACT}">')

    resolver = UnpaywallResolver(email="", transport=httpx.MockTransport(handler))
    paper = Paper(title="A paper")
    try:
        await resolver._discover_landing_page(paper, "https://publisher.example.org/paper")
        assert paper.abstract == ABSTRACT
        paper.abstract = "Existing trusted metadata."
        await resolver._discover_landing_page(paper, "https://publisher.example.org/paper")
        assert paper.abstract == "Existing trusted metadata."
    finally:
        await resolver.aclose()


def test_abstract_container_collects_nested_text_without_site_navigation():
    parser = _PdfLinks("https://publisher.example.org/paper")
    parser.feed(
        f'<nav>Subscribe to our journal</nav><span class="moreAbstract">{ABSTRACT}<b> Results.</b></span><footer>Terms</footer>'
    )
    assert parser.abstract == ABSTRACT + " Results."
    assert "Subscribe" not in parser.abstract and "Terms" not in parser.abstract


def test_inline_pdf_abstract_stops_before_introduction():
    text = f"# Title\n\nAbstract—{ABSTRACT}\n\nI. INTRODUCTION\nUnrelated body text."
    assert extract_abstract(text) == ABSTRACT
    assert extract_abstract("# Paper\nNo abstract section exists here.") is None


def test_real_architecture_description_with_honest_gap_is_not_a_missing_field():
    assert has_substantive_description(
        "终端、边缘、云三层执行位置；材料中未明确说明更细的编排机制。"
    )
    assert not has_substantive_description("材料中未明确说明。")
    assert not has_substantive_description("")
    assert (
        extract_abstract(
            f"Authors. Keywords: offloading. Abstract: {ABSTRACT}\n\n1 INTRODUCTION\nBody."
        )
        == ABSTRACT
    )


@pytest.mark.asyncio
async def test_model_selects_exact_source_by_id_without_retyping_pdf_glyphs():
    class Client:
        async def complete(self, messages, **kwargs):
            assert "[E001]" in messages[-1]["content"]
            return json.dumps({"method": "Q-learning", "evidence": [{"source_id": "E001"}]})

    paper = Paper(title="Paper", abstract=ABSTRACT)
    summary = await LlmSummarizer(chat_client=Client()).summarize(IDEA, paper)
    assert summary.generation_method == "model"
    assert summary.evidence[0].quote == ABSTRACT


@pytest.mark.asyncio
async def test_unknown_evidence_id_never_becomes_valid_source_evidence():
    class Client:
        async def complete(self, messages, **kwargs):
            return json.dumps({"method": "Invented method", "evidence": [{"source_id": "E999"}]})

    summary = await LlmSummarizer(chat_client=Client()).summarize(
        IDEA, Paper(title="Paper", abstract=ABSTRACT)
    )
    assert summary.generation_method == "extractive"
    assert "Invented method" not in summary.method


@pytest.mark.asyncio
async def test_llm_plus_dictionary_concepts_do_not_explode_boolean_request_budget():
    from paper_agent.adapters._concept_extraction import ExtractedConcepts
    from paper_agent.adapters.query_planner import HeuristicQueryPlanner

    class Extractor:
        def extract(self, text):
            return ExtractedConcepts(
                concepts=[
                    "task offloading",
                    "edge computing",
                    "resource allocation",
                    "cloud computing",
                    "resource scheduling",
                    "6G networks",
                ]
            )

    plan = await HeuristicQueryPlanner(concept_extractor=Extractor(), expand_queries=True).plan(
        IDEA, 15
    )
    assert len(plan.queries) <= 48
    for source in ("arxiv", "europe_pmc"):
        assert sum(q.source.value == source for q in plan.queries) <= 10


@pytest.mark.asyncio
async def test_unrecognized_method_heading_does_not_hide_explicit_proposal(tmp_path):
    text = f"# Paper\n\nAbstract—{ABSTRACT}\n\nIV. DETERMINISTIC OFFLOADING\n{ABSTRACT}"
    path = tmp_path / "paper.md"
    path.write_text(text)
    paper = Paper(title="Task offloading")
    paper.fulltext.markdown_path = str(path)
    summary = await LlmSummarizer().summarize(IDEA, paper)
    assert "Q-learning" in summary.method
    assert paper.abstract == ABSTRACT
    assert summary.generation_method == "extractive"
    assert any(ref.quote and quote_in_source(ref.quote, text) for ref in summary.evidence)


@pytest.mark.asyncio
async def test_fabricated_llm_quote_is_rejected_and_falls_back_to_real_source():
    class Client:
        calls = 0

        async def complete(self, messages, **kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "method": "Invented claim",
                    "evidence": [{"quote": "A fabricated result not in the paper."}],
                }
            )

    client = Client()
    paper = Paper(title="Task offloading", abstract=ABSTRACT)
    summary = await LlmSummarizer(chat_client=client).summarize(IDEA, paper)
    assert client.calls == 2
    assert summary.generation_method == "extractive"
    assert "Invented" not in summary.method
    assert all(ref.quote and quote_in_source(ref.quote, ABSTRACT) for ref in summary.evidence)


@pytest.mark.asyncio
async def test_valid_model_comparison_fields_and_source_quote_survive():
    class Client:
        async def complete(self, messages, **kwargs):
            assert "resource_constraints" in messages[-1]["content"]
            return json.dumps(
                {
                    "method": "Q-learning",
                    "architecture": "edge-cloud",
                    "resource_allocation": "Q-learning task offloading",
                    "resource_constraints": ["task deadlines"],
                    "evaluation_metrics": ["tasks meeting deadlines"],
                    "evidence": [{"quote": ABSTRACT}],
                }
            )

    summary = await LlmSummarizer(chat_client=Client()).summarize(
        IDEA, Paper(title="Paper", abstract=ABSTRACT)
    )
    assert summary.generation_method == "model"
    assert summary.architecture == "edge-cloud"
    assert summary.resource_constraints == ["task deadlines"]


@pytest.mark.asyncio
async def test_fallback_report_covers_all_papers_and_labels_metadata_only_instead_of_padding():
    papers = [Paper(title=f"Paper {i}", abstract=ABSTRACT if i < 4 else None) for i in range(5)]
    summaries = [await LlmSummarizer().summarize(IDEA, p) for p in papers]
    report = await ShortReviewWriter().write(IDEA, papers, summaries)
    assert report.generation_method == "extractive" and report.quality_notes
    assert "未配置模型" in report.markdown
    assert "仅题录，无实质性证据" in report.markdown
    assert "架构/层级" in report.markdown and "分配与卸载策略" in report.markdown
    assert all(p.title in report.markdown for p in papers)
    assert "当前材料未涵盖" not in report.executive_summary  # No repeated generic filler.


def test_planner_reuses_local_summary_key_but_explicit_empty_argument_disables_it(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setenv("PAPER_AGENT_LLM_API_KEY", "")
    assert LLMConceptExtractor().api_key == "fixture-only"
    assert LLMConceptExtractor(api_key="").active is False


def test_repeated_public_queries_use_cache_without_mixing_keys_or_parameters(monkeypatch):
    fetch = Mock(side_effect=lambda *a, **k: Mock(status_code=200, json=lambda: {"data": [1]}))
    monkeypatch.setattr(http.requests, "get", fetch)
    monkeypatch.setattr(http, "_host_gate", lambda *a: None)
    url = "https://api2.openreview.net/notes/search"
    first = http.http_get_json(
        url, params={"query": "offloading"}, headers={"x-api-key": "fixture-a"}
    )
    first["data"].append(2)
    again = http.http_get_json(
        url, params={"query": "offloading"}, headers={"x-api-key": "fixture-a"}
    )
    assert again == {"data": [1]} and fetch.call_count == 1
    http.http_get_json(url, params={"query": "offloading"}, headers={"x-api-key": "fixture-b"})
    http.http_get_json(url, params={"query": "other"}, headers={"x-api-key": "fixture-a"})
    assert fetch.call_count == 3
    assert all("fixture" not in key for key in http._RESPONSE_CACHE)


def test_system_trust_store_keeps_certificate_and_hostname_verification_enabled():
    context = verified_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_semantic_scholar_date_sort_uses_bulk_and_locally_caps_large_response(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return {"total": 1000, "data": [{"paperId": str(i)} for i in range(1000)]}

    monkeypatch.setattr("paper_agent.adapters.semantic_scholar_retriever.http_get_json", get)
    payload = SemanticScholarRetriever()._fetch("offloading", 3, sort_by="publication_date")
    assert calls[0][0].endswith("/paper/search/bulk")
    assert "limit" not in calls[0][1]
    assert len(payload["data"]) == 3
    SemanticScholarRetriever()._fetch("offloading", 3)
    assert calls[1][0].endswith("/paper/search") and calls[1][1]["limit"] == 3
