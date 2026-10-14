import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from paper_agent.adapters.academic_ranker import (
    AcademicEvidenceVerifier,
    AcademicFinalRanker,
    AcademicIntentRewriter,
    AcademicPreRanker,
    AcademicRankingConfig,
    BgeReranker,
)
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings
from paper_agent.domain.models import (
    FullTextAsset,
    Paper,
    PipelineRequest,
    ProcessingStatus,
    ResearchIdea,
    Venue,
)


class AcademicRankingTest(unittest.TestCase):
    def setUp(self):
        self.config = AcademicRankingConfig(reference_year=2026)
        self.idea = ResearchIdea(text="graph neural networks for recommendation")

    def test_relevance_beats_newness_venue_and_citations(self):
        relevant = Paper(title="Graph neural networks for recommendation", year=2018)
        unrelated = Paper(
            title="Quantum optical hardware",
            year=2026,
            citation_count=10000,
            venue=Venue(name="Preferred Venue"),
        )
        idea = self.idea.model_copy(update={"preferred_venues": ["Preferred Venue"]})
        for ranker in [AcademicPreRanker(self.config), AcademicFinalRanker(self.config)]:
            result = asyncio.run(ranker.rank(idea, [unrelated, relevant], 1))
            self.assertEqual(result[0].paper_id, relevant.paper_id)
        self.assertIsNone(relevant.scores.final)

    def test_pre_pool_and_final_top_k_are_distinct(self):
        papers = [Paper(title=f"Graph neural networks recommendation {i}") for i in range(5)]
        pool = asyncio.run(AcademicPreRanker(self.config).rank(self.idea, papers, 4))
        result = asyncio.run(AcademicFinalRanker(self.config).rank(self.idea, pool, 2))
        self.assertEqual(len(pool), 4)
        self.assertEqual(len(result), 2)
        self.assertEqual(pool[0].scores.formula_version, "academic-v1-pre")
        self.assertEqual(result[0].scores.formula_version, "academic-v1-final")
        self.assertIn("lexical-fallback", result[0].scores.explanation)
        self.assertIsNone(result[0].scores.reranker)

    def test_intent_preserves_original_and_structured_constraints(self):
        idea = ResearchIdea(
            text='最新 graph neural networks 开源代码 exclude "hardware"',
            required_terms=["graph"],
            ranking_focus=["custom"],
        )
        intent = asyncio.run(AcademicIntentRewriter().rewrite(idea))
        self.assertEqual(intent.text, idea.text)
        self.assertEqual(intent.required_terms, ["graph"])
        self.assertEqual(intent.excluded_terms, ["hardware"])
        self.assertTrue({"custom", "code", "recency"}.issubset(intent.ranking_focus))
        self.assertEqual(idea.excluded_terms, [])

    def test_hard_constraints_and_retraction_screening(self):
        idea = self.idea.model_copy(
            update={"required_terms": ["graph"], "excluded_terms": ["hardware"]}
        )
        papers = [
            Paper(title="Graph recommendation"),
            Paper(title="Graph hardware recommendation"),
            Paper(title="Graph recommendation withdrawn", is_retracted=True),
            Paper(title="Neural recommendation"),
        ]
        pool = asyncio.run(AcademicPreRanker(self.config).rank(idea, papers, 10))
        self.assertEqual(len(pool), 2)
        final = asyncio.run(AcademicFinalRanker(self.config).rank(idea, pool, 10))
        self.assertEqual([p.title for p in final], ["Graph recommendation"])

    def test_actual_markdown_can_supply_required_concept(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text(
                "Graph neural networks for recommendation with cold-start learning.",
                encoding="utf-8",
            )
            paper = Paper(
                title="Graph neural networks for recommendation",
                fulltext=FullTextAsset(markdown_path=str(path), parse_quality=0.9),
            )
            idea = self.idea.model_copy(update={"required_terms": ["cold-start"]})
            verified = asyncio.run(AcademicEvidenceVerifier().verify(idea, paper))
            self.assertEqual(verified.scores.evidence_coverage, 0.9)
            result = asyncio.run(AcademicFinalRanker(self.config).rank(idea, [verified], 1))
            self.assertEqual(len(result), 1)
            self.assertIn("evidence=markdown", result[0].scores.explanation)

    def test_nonexistent_markdown_does_not_count_as_fulltext(self):
        paper = Paper(
            title="Graph neural networks",
            abstract="Recommendation using graph neural networks",
            fulltext=FullTextAsset(markdown_path="nonexistent/paper.md", parse_quality=1),
        )
        verified = asyncio.run(AcademicEvidenceVerifier().verify(self.idea, paper))
        self.assertEqual(verified.scores.evidence_coverage, 0.35)
        self.assertIn("source=abstract", verified.scores.explanation)

    def test_reranker_scores_drive_final_order(self):
        ranker = AcademicFinalRanker(self.config)
        papers = [
            Paper(title="Graph neural networks recommendation A"),
            Paper(title="Graph neural networks recommendation B"),
        ]
        with patch.object(ranker.backend, "predict", return_value=([0.1, 0.9], "test-model")):
            result = asyncio.run(ranker.rank(self.idea, papers, 2))
        self.assertEqual(result[0].paper_id, papers[1].paper_id)
        self.assertEqual(result[0].scores.reranker, 0.9)

    def test_backend_normalizes_logits_and_loads_once(self):
        encoder = MagicMock()
        encoder.return_value.predict.return_value = [-1000.0, 1000.0]
        modules = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Identity=lambda: "identity")),
            "sentence_transformers": SimpleNamespace(CrossEncoder=encoder),
        }
        backend = BgeReranker(AcademicRankingConfig(model_name="local-test-model"))
        with patch.dict(sys.modules, modules):
            scores, _ = backend.predict("query", ["first", "second"])
            backend.predict("query", ["first", "second"])
        encoder.assert_called_once()
        self.assertTrue(encoder.call_args.kwargs["local_files_only"])
        self.assertLess(scores[0], 0.001)
        self.assertGreater(scores[1], 0.999)

    def test_failed_model_falls_back_without_repeated_loading(self):
        encoder = MagicMock(side_effect=OSError("model unavailable"))
        modules = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Identity=lambda: "identity")),
            "sentence_transformers": SimpleNamespace(CrossEncoder=encoder),
        }
        backend = BgeReranker(AcademicRankingConfig(model_name="missing-model"))
        with patch.dict(sys.modules, modules):
            scores, reason = backend.predict("query", ["passage"])
            backend.predict("query", ["passage"])
        self.assertIsNone(scores)
        self.assertIn("reranker failed: OSError", reason)
        encoder.assert_called_once()

    def test_ranking_mode_runs_entire_pipeline(self):
        agent = build_agent(Settings(app_env="ranking"))
        result = asyncio.run(
            agent.research(PipelineRequest(idea=self.idea, top_k=2, pre_rank_pool_size=3))
        )
        self.assertIn(result.status, {ProcessingStatus.SUCCEEDED, ProcessingStatus.PARTIAL})
        self.assertEqual(len(result.papers), 2)
        self.assertTrue(all(p.scores.formula_version == "academic-v1-final" for p in result.papers))

    def test_date_range_uses_exact_date_when_available(self):
        from datetime import date

        idea = self.idea.model_copy(update={"date_from": date(2026, 7, 1)})
        papers = [
            Paper(title="Graph neural networks early", publication_date=date(2026, 1, 1)),
            Paper(title="Graph neural networks late", publication_date=date(2026, 8, 1)),
        ]
        result = asyncio.run(AcademicPreRanker(self.config).rank(idea, papers, 2))
        self.assertEqual([p.title for p in result], [papers[1].title])

    def test_empty_input_and_invalid_limit(self):
        for ranker in [AcademicPreRanker(self.config), AcademicFinalRanker(self.config)]:
            self.assertEqual(asyncio.run(ranker.rank(self.idea, [], 1)), [])
            with self.assertRaises(ValueError):
                asyncio.run(ranker.rank(self.idea, [], 0))
