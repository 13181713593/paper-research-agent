"""Main-agent facade shared by MCP and Streamlit.

The LLM host sees one coarse-grained research capability. This facade is the
single composition boundary: UI and MCP code must not reproduce pipeline logic.
"""

from __future__ import annotations

from paper_agent.application.pipeline import EventSink, ResearchPipeline
from paper_agent.domain.models import (
    PipelineContract,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
)


class PaperResearchAgent:
    """Validate a request, invoke the deterministic workflow, return audit data."""

    def __init__(self, pipeline: ResearchPipeline, version: str = "0.2.0") -> None:
        self._pipeline = pipeline
        self._version = version

    async def research(
        self, request: PipelineRequest, event_sink: EventSink | None = None
    ) -> PipelineResult:
        """Run the complete research workflow.

        Input:
            A validated ``PipelineRequest`` containing the research idea and
            candidate/full-text selection policy.
        Output:
            ``PipelineResult`` with Top-k papers, summaries, report, stage
            timings, provenance, warnings, and partial/failure status.
        """
        return await self._pipeline.run(request, event_sink=event_sink)

    def contract(self) -> PipelineContract:
        """Expose stable orchestration semantics to hosts and integration tests."""
        return PipelineContract(
            version=self._version,
            stages=[
                PipelineStage.QUERY_PLANNED,
                PipelineStage.RETRIEVED,
                PipelineStage.DEDUPED,
                PipelineStage.PRE_RANKED,
                PipelineStage.FULLTEXT_PROCESSED,
                PipelineStage.VERIFIED,
                PipelineStage.FINAL_RANKED,
                PipelineStage.SUMMARIZED,
                PipelineStage.REPORTED,
            ],
            notes=[
                "Pre-ranking selects a full-text candidate pool; it is not final Top-k.",
                "Final ranking occurs only after full-text or explicit abstract fallback verification.",
            ],
        )

