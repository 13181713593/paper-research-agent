"""Offline unit tests for Group 1 B-side components.

All network calls are mocked; these tests run without internet access and
cover the acceptance criteria in docs/API_CONTRACTS.md: empty results, field
missing, illegal dates, rate limits, timeouts.
"""

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from unittest import mock

import requests

from paper_agent.adapters._concept_extraction import HeuristicConceptExtractor
from paper_agent.adapters._retriever_utils import (
    ProviderRateLimitedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    http_get_json,
    normalize_doi,
    parse_date,
    parse_year,
    restore_abstract_from_inverted_index,
)
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.llm_concept_extractor import LLMConceptExtractor
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.domain.models import PaperSource, ResearchIdea, SearchPlan, SearchQuery

IDEA_ZH = "用强化学习提升小语言模型的数学推理能力"


class ConceptExtractionTest(unittest.TestCase):
    def test_chinese_idea_maps_to_english_terms(self):
        extracted = HeuristicConceptExtractor().extract(IDEA_ZH)
        self.assertIn("reinforcement learning", extracted.concepts)
        self.assertIn("mathematical reasoning", extracted.concepts)
        self.assertIn("small language model", extracted.concepts)

    def test_longest_match_wins(self):
        extracted = HeuristicConceptExtractor().extract("深度强化学习")
        self.assertIn("deep reinforcement learning", extracted.concepts)
        self.assertNotIn("reinforcement learning", extracted.concepts)

    def test_english_idea_keeps_tokens_without_stopwords(self):
        extracted = HeuristicConceptExtractor().extract(
            "improving mathematical reasoning with self-training"
        )
        self.assertIn("mathematical", extracted.concepts)
        self.assertNotIn("with", extracted.concepts)

    def test_synonyms_follow_concepts(self):
        extracted = HeuristicConceptExtractor().extract(IDEA_ZH)
        self.assertTrue(set(extracted.synonyms) & {"RL", "policy optimization", "LLM"})

    def test_subsumed_concepts_are_dropped(self):
        extracted = HeuristicConceptExtractor().extract(IDEA_ZH)
        self.assertIn("mathematical reasoning", extracted.concepts)
        self.assertNotIn("reasoning", extracted.concepts)


class QueryPlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = HeuristicQueryPlanner()

    def test_plan_has_one_query_per_source(self):
        plan = asyncio.run(self.planner.plan(ResearchIdea(text=IDEA_ZH), 25))
        sources = [q.source for q in plan.queries]
        self.assertEqual(
            set(sources),
            {
                PaperSource.ARXIV,
                PaperSource.OPENALEX,
                PaperSource.CROSSREF,
                PaperSource.SEMANTIC_SCHOLAR,
                PaperSource.EUROPE_PMC,
                PaperSource.OPENREVIEW,
            },
        )
        for q in plan.queries:
            self.assertEqual(q.limit, 25)
            self.assertTrue(q.query)

    def test_required_terms_included_in_query(self):
        idea = ResearchIdea(text=IDEA_ZH, required_terms=["curriculum learning"])
        plan = asyncio.run(self.planner.plan(idea, 10))
        for q in plan.queries:
            if q.source == PaperSource.ARXIV:
                # arXiv queries carry only the top concept; required terms
                # are enforced via the filter the adapter ANDs in.
                self.assertEqual(q.filters.get("required_terms"), ["curriculum learning"])
            else:
                self.assertIn("curriculum learning", q.query)

    def test_inclusion_criteria_mention_required_terms(self):
        idea = ResearchIdea(text=IDEA_ZH, required_terms=["GRPO"])
        plan = asyncio.run(self.planner.plan(idea, 10))
        self.assertTrue(any("GRPO" in c for c in plan.inclusion_criteria))

    def test_excluded_terms_flow_into_exclusion(self):
        idea = ResearchIdea(text=IDEA_ZH, excluded_terms=["survey"])
        plan = asyncio.run(self.planner.plan(idea, 10))
        self.assertIn("survey", plan.exclusion_criteria)

    def test_filters_follow_shared_retrieval_protocol(self):
        idea = ResearchIdea(text=IDEA_ZH, excluded_terms=["survey"])
        plan = asyncio.run(self.planner.plan(idea, 10))
        for q in plan.queries:
            self.assertEqual(q.filters.get("excluded_terms"), ["survey"])
            self.assertNotIn("synonyms", q.filters)
            self.assertLessEqual(
                set(q.filters), {"date_from", "date_to", "excluded_terms", "required_terms"}
            )

    def test_no_filters_when_no_exclusions(self):
        plan = asyncio.run(self.planner.plan(ResearchIdea(text=IDEA_ZH), 10))
        for q in plan.queries:
            self.assertEqual(q.filters, {})


class LLMConceptExtractorTest(unittest.TestCase):
    def test_inactive_without_key_uses_heuristic(self):
        # Explicitly blank the key: earlier tests may call build_agent(),
        # whose load_local_env() imports the developer's .env into this
        # process. This test must stay hermetic and offline.
        with mock.patch.dict("os.environ", {"PAPER_AGENT_LLM_API_KEY": ""}):
            extractor = LLMConceptExtractor(api_key=None)
            extracted = extractor.extract(IDEA_ZH)
            self.assertIn("reinforcement learning", extracted.concepts)
            self.assertEqual(extractor.last_mode, "heuristic")

    def test_llm_failure_degrades_to_heuristic(self):
        extractor = LLMConceptExtractor(api_key="sk-test")
        with mock.patch("paper_agent.adapters.llm_concept_extractor.requests.post") as post:
            post.side_effect = requests.ConnectionError("down")
            extracted = extractor.extract(IDEA_ZH)
        self.assertIn("reinforcement learning", extracted.concepts)
        self.assertEqual(extractor.last_mode, "heuristic")

    def test_llm_success_parses_concepts(self):
        extractor = LLMConceptExtractor(api_key="sk-test")
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "concepts": ["GRPO", "process reward model"],
                                "synonyms": ["RL"],
                            }
                        )
                    }
                }
            ]
        }
        with mock.patch("paper_agent.adapters.llm_concept_extractor.requests.post") as post:
            post.return_value = response
            extracted = extractor.extract(IDEA_ZH)
        self.assertEqual(extracted.concepts, ["GRPO", "process reward model"])
        self.assertEqual(extractor.last_mode, "llm")

    def test_llm_garbage_json_degrades(self):
        extractor = LLMConceptExtractor(api_key="sk-test")
        response = mock.Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "no json here"}}]}
        with mock.patch("paper_agent.adapters.llm_concept_extractor.requests.post") as post:
            post.return_value = response
            extractor.extract(IDEA_ZH)
        self.assertEqual(extractor.last_mode, "heuristic")


class HttpUtilTest(unittest.TestCase):
    def test_timeout_raises_typed_error(self):
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.side_effect = requests.Timeout("t")
            with self.assertRaises(ProviderTimeoutError):
                http_get_json("http://example.com")

    def test_persistent_429_raises_rate_limited(self):
        response = mock.Mock(status_code=429)
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.return_value = response
            with self.assertRaises(ProviderRateLimitedError):
                http_get_json("http://example.com", max_retries=1)

    def test_retry_after_seconds_header_is_honored(self):
        limited = mock.Mock(status_code=429, headers={"Retry-After": "7"})
        ok = mock.Mock(status_code=200)
        ok.json.return_value = {"data": []}
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.side_effect = [limited, ok]
            clock = [1000.0]
            with (
                mock.patch(
                    "paper_agent.adapters._retriever_utils.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
                mock.patch(
                    "paper_agent.adapters._retriever_utils.time.sleep",
                    side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay),
                ) as sleep,
            ):
                result = http_get_json("http://example.com", max_retries=2)
        self.assertEqual(result, {"data": []})
        sleep.assert_called_once_with(7.0)

    def test_retry_after_http_date_is_honored(self):
        future = datetime.now(UTC) + timedelta(minutes=2)
        http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        limited = mock.Mock(status_code=429, headers={"Retry-After": http_date})
        ok = mock.Mock(status_code=200)
        ok.json.return_value = {}
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.side_effect = [limited, ok]
            clock = [1000.0]
            with (
                mock.patch(
                    "paper_agent.adapters._retriever_utils.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
                mock.patch(
                    "paper_agent.adapters._retriever_utils.time.sleep",
                    side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay),
                ) as sleep,
            ):
                http_get_json(
                    "http://example.com", max_retries=2, total_timeout=150, max_retry_after=120
                )
        waited = sleep.call_args[0][0]
        self.assertGreater(waited, 60)
        self.assertLessEqual(waited, 120)

    def test_500_retries_then_raises_unavailable(self):
        response = mock.Mock(status_code=500)
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.return_value = response
            with self.assertRaises(ProviderUnavailableError):
                http_get_json("http://example.com", max_retries=1)

    def test_invalid_json_raises_response_error(self):
        response = mock.Mock(status_code=200)
        response.json.side_effect = ValueError("bad")
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.return_value = response
            with self.assertRaises(ProviderResponseError):
                http_get_json("http://example.com")

    def test_normalize_doi_variants(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1234/abc"), "10.1234/abc")
        self.assertEqual(normalize_doi("10.1234/abc"), "10.1234/abc")
        self.assertIsNone(normalize_doi(None))

    def test_parse_date_illegal_returns_none(self):
        self.assertIsNone(parse_date("not-a-date"))
        self.assertIsNone(parse_date(None))
        self.assertEqual(parse_date("2024-03-01").year, 2024)

    def test_parse_year_bounds(self):
        self.assertIsNone(parse_year("garbage"))
        self.assertIsNone(parse_year(1500))
        self.assertIsNone(parse_year(2500))
        self.assertEqual(parse_year("2024"), 2024)

    def test_inverted_index_restoration(self):
        restored = restore_abstract_from_inverted_index({"word": [0, 2], "next": [1]})
        self.assertEqual(restored, "word next word")
        self.assertIsNone(restore_abstract_from_inverted_index(None))


CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "https://doi.org/10.1000/demo",
                "title": ["Deep RL for Mathematical Reasoning"],
                "author": [
                    {
                        "given": "Alex",
                        "family": "Chen",
                        "ORCID": "https://orcid.org/0000-0001-0000-0000",
                        "affiliation": [{"name": "Example University"}],
                    },
                ],
                "container-title": ["Journal of Demo"],
                "issued": {"date-parts": [[2024, 5, 3]]},
                "abstract": "<p>We study <i>reasoning</i>.</p>",
                "is-referenced-by-count": 42,
                "URL": "https://doi.org/10.1000/demo",
                "type": "journal-article",
            },
            {
                "DOI": "10.1000/empty",
                "title": ["No authors, illegal date"],
                "issued": {"date-parts": [["garbage"]]},
            },
        ]
    }
}

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/openalex",
            "display_name": "Group Relative Policy Optimization",
            "authorships": [
                {
                    "author": {
                        "display_name": "Taylor Li",
                        "orcid": "https://orcid.org/0000-0002-0000-0000",
                    },
                    "institutions": [{"display_name": "Demo Institute"}],
                },
            ],
            "publication_year": 2024,
            "publication_date": "2024-02-01",
            "primary_location": {"source": {"display_name": "arXiv"}},
            "cited_by_count": 7,
            "open_access": {"is_oa": True, "oa_url": "https://example.org/oa.pdf"},
            "abstract_inverted_index": {"We": [0], "study": [1], "GRPO": [2]},
        },
        {
            "id": "https://openalex.org/W456",
            "display_name": "No abstract here",
            "publication_year": 2023,
        },
    ]
}

S2_PAYLOAD = {
    "data": [
        {
            "paperId": "abc123",
            "title": "DAPO: An Open-Source System",
            "abstract": "We open source everything.",
            "year": 2025,
            "publicationDate": "2025-03-10",
            "authors": [{"name": "Robin Zhang"}, {"name": ""}],
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.1000/dapo", "ArXiv": "2503.14476"},
            "citationCount": 3,
            "openAccessPdf": {"url": "https://example.org/dapo.pdf"},
        },
        {"paperId": "def456", "title": "", "year": 2024},
    ]
}


def _plan_for(source: PaperSource) -> SearchPlan:
    return SearchPlan(
        intent_summary="demo",
        queries=[SearchQuery(source=source, query="reinforcement learning reasoning", limit=5)],
    )


class RetrieverMappingTest(unittest.TestCase):
    def test_crossref_maps_fields_and_strips_jats(self):
        retriever = CrossrefRetriever()
        papers = retriever._to_papers(CROSSREF_PAYLOAD)
        self.assertEqual(len(papers), 2)
        first = papers[0]
        self.assertEqual(first.title, "Deep RL for Mathematical Reasoning")
        self.assertEqual(first.doi, "10.1000/demo")
        self.assertEqual(first.authors[0].name, "Alex Chen")
        self.assertIn("Example University", first.institutions)
        self.assertEqual(first.year, 2024)
        self.assertEqual(first.abstract, "We study reasoning .")
        self.assertEqual(first.citation_count, 42)
        self.assertIn("@article{", first.bibtex)
        self.assertEqual(first.source_records[0].source, PaperSource.CROSSREF)

    def test_crossref_illegal_date_becomes_none_not_crash(self):
        retriever = CrossrefRetriever()
        second = retriever._to_papers(CROSSREF_PAYLOAD)[1]
        self.assertIsNone(second.publication_date)

    def test_openalex_restores_inverted_abstract(self):
        retriever = OpenAlexRetriever()
        papers = retriever._to_papers(OPENALEX_PAYLOAD)
        self.assertEqual(len(papers), 2)
        first = papers[0]
        self.assertEqual(first.abstract, "We study GRPO")
        self.assertEqual(first.openalex_id, "W123")
        self.assertTrue(first.is_open_access)
        self.assertEqual(first.authors[0].orcid, "0000-0002-0000-0000")
        second = papers[1]
        self.assertIsNone(second.abstract)

    def test_semantic_scholar_maps_external_ids(self):
        retriever = SemanticScholarRetriever()
        papers = retriever._to_papers(S2_PAYLOAD)
        self.assertEqual(len(papers), 1)
        first = papers[0]
        self.assertEqual(first.title, "DAPO: An Open-Source System")
        self.assertEqual(first.doi, "10.1000/dapo")
        self.assertEqual(first.arxiv_id, "2503.14476")
        self.assertEqual(first.semantic_scholar_id, "abc123")
        self.assertEqual(first.venue.name, "NeurIPS")
        self.assertIn("Robin Zhang", [a.name for a in first.authors])

    def test_semantic_scholar_empty_pdf_url_becomes_none(self):
        # Real S2 records can carry openAccessPdf with an empty url string;
        # the contract requires None instead of "" (URL validation fails).
        retriever = SemanticScholarRetriever()
        payload = {
            "data": [
                {
                    "paperId": "xyz789",
                    "title": "Empty OA URL Record",
                    "year": 2024,
                    "authors": [{"name": "Taylor Li"}],
                    "openAccessPdf": {"url": ""},
                }
            ]
        }
        paper = retriever._to_papers(payload)[0]
        self.assertIsNone(paper.open_access_url)
        self.assertFalse(paper.is_open_access)

    def test_openalex_key_sent_in_header_not_query_parameters(self):
        retriever = OpenAlexRetriever(api_key="fixture-openalex-key")
        with mock.patch("paper_agent.adapters.openalex_retriever.http_get_json") as get:
            get.return_value = {}
            retriever._fetch("scientific paper", 5)
        self.assertEqual(
            get.call_args.kwargs["headers"],
            {
                "Authorization": "Bearer fixture-openalex-key",
            },
        )
        self.assertNotIn("api_key", get.call_args.kwargs["params"])
        self.assertNotIn("fixture-openalex-key", get.call_args.args[0])

    def test_openalex_explicit_empty_key_keeps_anonymous_calls(self):
        retriever = OpenAlexRetriever(api_key="")
        with mock.patch("paper_agent.adapters.openalex_retriever.http_get_json") as get:
            get.return_value = {}
            retriever._fetch("scientific paper", 5)
        self.assertNotIn("headers", get.call_args.kwargs)

    def test_openalex_empty_oa_url_becomes_none(self):
        retriever = OpenAlexRetriever()
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W789",
                    "display_name": "Empty OA URL Record",
                    "publication_year": 2024,
                    "open_access": {"is_oa": True, "oa_url": ""},
                }
            ]
        }
        paper = retriever._to_papers(payload)[0]
        self.assertIsNone(paper.open_access_url)

    def test_retriever_returns_empty_when_plan_lacks_its_source(self):
        for retriever in (CrossrefRetriever(), OpenAlexRetriever(), SemanticScholarRetriever()):
            plan = _plan_for(PaperSource.ARXIV)
            self.assertEqual(asyncio.run(retriever.search(plan)), [])

    def test_search_empty_payload_returns_empty_list(self):
        retriever = CrossrefRetriever()
        response = mock.Mock(status_code=200)
        response.json.return_value = {"message": {"items": []}}
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.return_value = response
            plan = _plan_for(PaperSource.CROSSREF)
            result = asyncio.run(retriever.search(plan))
        self.assertEqual(result, [])

    def test_search_404_raises_instead_of_silent_empty(self):
        retriever = CrossrefRetriever()
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.return_value = mock.Mock(status_code=404, text="not found")
            plan = _plan_for(PaperSource.CROSSREF)
            with self.assertRaises(ProviderResponseError):
                asyncio.run(retriever.search(plan))

    def test_search_provider_error_propagates_typed(self):
        retriever = OpenAlexRetriever()
        with mock.patch("paper_agent.adapters._retriever_utils.requests.get") as get:
            get.side_effect = requests.Timeout("t")
            plan = _plan_for(PaperSource.OPENALEX)
            with self.assertRaises(ProviderTimeoutError):
                asyncio.run(retriever.search(plan))


_EPMC_RECORD = {
    "title": "A context-aware temporal modeling for automatic sleep staging",
    "authorString": "Vakili AA, Jahanshiri S, Salimi-Badr A.",
    "journalTitle": "PLoS ONE",
    "pubYear": "2026",
    "firstPublicationDate": "2026-09-30",
    "doi": "10.1371/journal.pone.0358241",
    "pmid": "42814731",
    "pmcid": "PMC13626375",
    "abstractText": "Automatic sleep staging is a critical task in healthcare.",
    "isOpenAccess": "Y",
    "citedByCount": "0",
}


class EuropePmcRetrieverTest(unittest.TestCase):
    @staticmethod
    def _payload(*records):
        return {"resultList": {"result": list(records)}}

    def test_record_maps_to_paper_fields(self):
        retriever = EuropePmcRetriever()
        paper = retriever._to_papers(self._payload(_EPMC_RECORD))[0]
        self.assertEqual(paper.title, _EPMC_RECORD["title"])
        self.assertEqual(len(paper.authors), 3)
        self.assertEqual(paper.authors[0].name, "Vakili AA")
        self.assertEqual(paper.year, 2026)
        self.assertEqual(str(paper.publication_date), "2026-09-30")
        self.assertEqual(paper.doi, "10.1371/journal.pone.0358241")
        self.assertEqual(paper.pmid, "42814731")
        self.assertEqual(paper.venue.name, "PLoS ONE")
        self.assertTrue(paper.is_open_access)
        self.assertEqual(paper.citation_count, 0)
        self.assertIn("PMC13626375", str(paper.landing_page_url))
        self.assertIn("journal.pone", paper.bibtex)
        self.assertEqual(paper.source_records[0].source, PaperSource.EUROPE_PMC)

    def test_record_without_title_is_skipped(self):
        retriever = EuropePmcRetriever()
        record = dict(_EPMC_RECORD)
        record["title"] = ""
        self.assertEqual(retriever._to_papers(self._payload(record)), [])

    def test_missing_optional_fields_stay_none(self):
        retriever = EuropePmcRetriever()
        record = {"title": "Minimal record"}
        paper = retriever._to_papers(self._payload(record))[0]
        self.assertEqual(paper.authors, [])
        self.assertIsNone(paper.doi)
        self.assertIsNone(paper.pmid)
        self.assertIsNone(paper.abstract)
        self.assertIsNone(paper.venue)
        self.assertIsNone(paper.landing_page_url)
        self.assertFalse(paper.is_open_access)

    def test_empty_payload_returns_empty_list(self):
        self.assertEqual(EuropePmcRetriever()._to_papers(None), [])
        self.assertEqual(EuropePmcRetriever()._to_papers({"resultList": {}}), [])

    def test_search_returns_empty_when_plan_lacks_its_source(self):
        plan = _plan_for(PaperSource.ARXIV)
        self.assertEqual(asyncio.run(EuropePmcRetriever().search(plan)), [])

    def test_planner_emits_query_for_europe_pmc(self):
        planner = HeuristicQueryPlanner()
        plan = asyncio.run(planner.plan(ResearchIdea(text=IDEA_ZH), 10))
        sources = {q.source for q in plan.queries}
        self.assertIn(PaperSource.EUROPE_PMC, sources)


if __name__ == "__main__":
    unittest.main()
