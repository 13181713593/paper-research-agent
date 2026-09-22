import asyncio
import unittest
from dataclasses import replace

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import PipelineRequest, ProcessingStatus, ResearchIdea


class FailingRetriever:
    source_name = "failing_demo_source"

    async def search(self, plan):
        raise TimeoutError("simulated provider timeout")


class PipelineTest(unittest.TestCase):
    def test_demo_pipeline_is_end_to_end(self):
        request = PipelineRequest(
            idea=ResearchIdea(text="multisource evidence-aware paper retrieval for academic research"),
            top_k=2,
            pre_rank_pool_size=3,
        )
        result = asyncio.run(ResearchPipeline(build_demo_dependencies()).run(request))
        self.assertIn(result.status, {ProcessingStatus.SUCCEEDED, ProcessingStatus.PARTIAL})
        self.assertEqual(len(result.papers), 2)
        self.assertEqual(len(result.summaries), 2)
        self.assertIsNotNone(result.report)
        self.assertGreaterEqual(len(result.events), 8)

    def test_invalid_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            PipelineRequest(
                idea=ResearchIdea(text="a sufficiently detailed research idea"),
                top_k=5,
                pre_rank_pool_size=4,
            )

    def test_one_source_can_fail_without_losing_successful_results(self):
        deps = build_demo_dependencies()
        deps = replace(deps, retrievers=[deps.retrievers[0], FailingRetriever()])
        request = PipelineRequest(
            idea=ResearchIdea(text="robust partial failure handling for academic search"),
            top_k=1,
            pre_rank_pool_size=2,
        )
        result = asyncio.run(ResearchPipeline(deps).run(request))
        self.assertEqual(result.status, ProcessingStatus.PARTIAL)
        self.assertEqual(len(result.papers), 1)
        self.assertTrue(any("failing_demo_source" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
