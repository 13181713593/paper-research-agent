"""Deterministic orchestration layer shared by MCP and Streamlit entry points."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from paper_agent.domain.models import (
    Paper,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    StageEvent,
)
from paper_agent.ports.contracts import (
    EvidenceVerifier,
    FinalRanker,
    FullTextProcessor,
    PaperMerger,
    PaperRetriever,
    PaperSummarizer,
    PreRanker,
    QueryPlanner,
    ReportWriter,
)

EventSink = Callable[[StageEvent], Awaitable[None]]


@dataclass(slots=True)
class PipelineDependencies:
    query_planner: QueryPlanner
    retrievers: Sequence[PaperRetriever]
    merger: PaperMerger
    pre_ranker: PreRanker
    fulltext_processor: FullTextProcessor
    verifier: EvidenceVerifier
    final_ranker: FinalRanker
    summarizer: PaperSummarizer
    report_writer: ReportWriter


class ResearchPipeline:
    def __init__(self, deps: PipelineDependencies, max_concurrency: int = 4):
        if len(deps.retrievers) < 2:
            raise ValueError("At least two independent retrievers are required")
        self.deps = deps
        self.max_concurrency = max_concurrency

    async def run(
        self, request: PipelineRequest, event_sink: EventSink | None = None
    ) -> PipelineResult:
        events: list[StageEvent] = []
        warnings: list[str] = []

        async def stage(
            name: PipelineStage,
            operation: Callable[[], Awaitable[object]],
            input_count: int | None = None,
        ) -> object:
            started_at = datetime.now(UTC)
            started = perf_counter()
            running = StageEvent(
                stage=name, status=ProcessingStatus.RUNNING, started_at=started_at,
                input_count=input_count,
            )
            if event_sink:
                await event_sink(running)
            try:
                value = await operation()
                output_count = len(value) if isinstance(value, (list, tuple)) else None
                event = StageEvent(
                    stage=name, status=ProcessingStatus.SUCCEEDED, started_at=started_at,
                    finished_at=datetime.now(UTC), input_count=input_count,
                    output_count=output_count, duration_ms=int((perf_counter() - started) * 1000),
                )
            except Exception as exc:
                event = StageEvent(
                    stage=name, status=ProcessingStatus.FAILED, started_at=started_at,
                    finished_at=datetime.now(UTC), input_count=input_count,
                    duration_ms=int((perf_counter() - started) * 1000),
                    message=str(exc), error_code=type(exc).__name__,
                )
                events.append(event)
                if event_sink:
                    await event_sink(event)
                raise
            events.append(event)
            if event_sink:
                await event_sink(event)
            return value

        try:
            plan = await stage(
                PipelineStage.QUERY_PLANNED,
                lambda: self.deps.query_planner.plan(request.idea, request.candidate_limit_per_source),
            )

            async def retrieve_all() -> list[Paper]:
                results = await asyncio.gather(
                    *(retriever.search(plan) for retriever in self.deps.retrievers),
                    return_exceptions=True,
                )
                papers: list[Paper] = []
                for retriever, result in zip(self.deps.retrievers, results, strict=True):
                    if isinstance(result, BaseException):
                        warnings.append(f"source {retriever.source_name} failed: {result}")
                    else:
                        papers.extend(result)
                if not papers:
                    raise RuntimeError("All paper sources failed or returned no results")
                return papers

            retrieved = await stage(PipelineStage.RETRIEVED, retrieve_all)
            deduped = await stage(
                PipelineStage.DEDUPED,
                lambda: self.deps.merger.merge_and_deduplicate(retrieved),
                len(retrieved),
            )
            pool = await stage(
                PipelineStage.PRE_RANKED,
                lambda: self.deps.pre_ranker.rank(
                    request.idea, deduped, request.pre_rank_pool_size
                ),
                len(deduped),
            )

            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def process_one(paper: Paper) -> Paper | None:
                async with semaphore:
                    try:
                        return await self.deps.fulltext_processor.process(paper)
                    # Adapter implementations come from independent groups and may
                    # wrap several third-party SDKs. This per-item boundary must
                    # convert any adapter failure into fallback/exclusion state.
                    except Exception as exc:  # noqa: BLE001
                        if request.allow_abstract_fallback and paper.abstract:
                            warnings.append(f"{paper.title}: full text unavailable; abstract fallback ({exc})")
                            return paper
                        warnings.append(f"{paper.title}: excluded because full text failed ({exc})")
                        return None

            async def process_pool() -> list[Paper]:
                processed = await asyncio.gather(*(process_one(p) for p in pool))
                return [p for p in processed if p is not None]

            processed = await stage(
                PipelineStage.FULLTEXT_PROCESSED, process_pool, len(pool)
            )
            if request.require_fulltext:
                processed = [p for p in processed if p.fulltext.markdown_path]
            if len(processed) < request.top_k:
                warnings.append(
                    f"Only {len(processed)} eligible papers remain for requested top_k={request.top_k}"
                )

            async def verify_all() -> list[Paper]:
                return list(await asyncio.gather(
                    *(self.deps.verifier.verify(request.idea, p) for p in processed)
                ))

            verified = await stage(PipelineStage.VERIFIED, verify_all, len(processed))
            final_papers = await stage(
                PipelineStage.FINAL_RANKED,
                lambda: self.deps.final_ranker.rank(request.idea, verified, request.top_k),
                len(verified),
            )

            async def summarize_all():
                return list(await asyncio.gather(
                    *(self.deps.summarizer.summarize(request.idea, p) for p in final_papers)
                ))

            summaries = await stage(
                PipelineStage.SUMMARIZED, summarize_all, len(final_papers)
            )
            report = await stage(
                PipelineStage.REPORTED,
                lambda: self.deps.report_writer.write(request.idea, final_papers, summaries),
                len(final_papers),
            )
            status = ProcessingStatus.PARTIAL if warnings else ProcessingStatus.SUCCEEDED
            return PipelineResult(
                status=status, stage=PipelineStage.REPORTED, request=request,
                papers=final_papers, summaries=summaries, report=report,
                events=events, warnings=warnings,
            )
        # Top-level application boundary: MCP/UI callers receive a structured
        # failed result with audit events instead of an untyped transport crash.
        except Exception as exc:  # noqa: BLE001
            warnings.append(str(exc))
            return PipelineResult(
                status=ProcessingStatus.FAILED, stage=PipelineStage.FAILED,
                request=request, events=events, warnings=warnings,
            )
