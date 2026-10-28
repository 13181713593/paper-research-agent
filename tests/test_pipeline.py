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
            idea=ResearchIdea(
                text="multisource evidence-aware paper retrieval for academic research"
            ),
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


def test_main_pipeline_bounds_both_processing_and_summary_calls():
    deps = build_demo_dependencies()

    class TrackedProcessor:
        active = peak = 0

        async def process(self, paper):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.01)
                return await original_processor.process(paper)
            finally:
                self.active -= 1

    class TrackedSummarizer:
        active = peak = 0

        async def summarize(self, idea, paper):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.01)
                return await original_summarizer.summarize(idea, paper)
            finally:
                self.active -= 1

    original_processor, original_summarizer = deps.fulltext_processor, deps.summarizer
    processor, summarizer = TrackedProcessor(), TrackedSummarizer()
    deps.fulltext_processor, deps.summarizer = processor, summarizer
    request = PipelineRequest(
        idea=ResearchIdea(text="scientific models"), top_k=2, pre_rank_pool_size=3
    )
    result = asyncio.run(ResearchPipeline(deps, max_concurrency=1).run(request))
    assert result.status != ProcessingStatus.FAILED, result.warnings
    assert len(result.summaries) == 2
    assert processor.peak == summarizer.peak == 1


def test_main_pipeline_accepts_larger_download_pool():
    request = PipelineRequest(
        idea=ResearchIdea(text="scientific models"), top_k=100, pre_rank_pool_size=200
    )
    assert request.top_k == 100 and request.pre_rank_pool_size == 200
    for concurrency in (0, 33):
        try:
            ResearchPipeline(build_demo_dependencies(), max_concurrency=concurrency)
        except ValueError:
            continue
        raise AssertionError("应拒绝不合法的并发预算")


if __name__ == "__main__":
    unittest.main()
