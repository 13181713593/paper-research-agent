"""Deterministic orchestration layer shared by MCP and Streamlit entry points."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic, perf_counter

from paper_agent.application.progress import output_count, output_snapshot, paper_snapshot
from paper_agent.domain.models import (
    Paper,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    StageEvent,
)
from paper_agent.domain.provider_messages import provider_message
from paper_agent.domain.publication_policy import is_recent, reserve_recent
from paper_agent.domain.request_budget import RequestBudget, active_budget
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
    def __init__(
        self,
        deps: PipelineDependencies,
        max_concurrency: int = 4,
        source_timeout_seconds: float = 45.0,
    ):
        if len(deps.retrievers) < 2:
            raise ValueError("At least two independent retrievers are required")
        if not 1 <= max_concurrency <= 32:
            raise ValueError("并发数必须在 1 至 32 之间。")
        self.deps = deps
        self.max_concurrency = max_concurrency
        if source_timeout_seconds <= 0:
            raise ValueError("source_timeout_seconds must be positive")
        self.source_timeout_seconds = source_timeout_seconds

    async def run(
        self, request: PipelineRequest, event_sink: EventSink | None = None
    ) -> PipelineResult:
        events: list[StageEvent] = []
        progress_events: list[StageEvent] = []
        warnings: list[str] = []
        metadata_papers: list[Paper] = []

        async def emit(event: StageEvent) -> None:
            # Running events drive the live view; terminal snapshots survive reruns/JSON export.
            if event.status != ProcessingStatus.RUNNING:
                progress_events.append(event)
                if event.scope == "stage":
                    events.append(event)
            if event_sink:
                await event_sink(event)

        async def stage(
            name: PipelineStage,
            operation: Callable[[], Awaitable[object]],
            input_count: int | None = None,
            component: object | None = None,
        ) -> object:
            started_at = datetime.now(UTC)
            started = perf_counter()
            warning_start = len(warnings)
            details = {"demo": type(component).__name__.startswith("Demo")} if component else {}
            running = StageEvent(
                stage=name,
                status=ProcessingStatus.RUNNING,
                started_at=started_at,
                input_count=input_count,
                details=details,
            )
            await emit(running)
            try:
                value = await operation()
                event = StageEvent(
                    stage=name,
                    status=ProcessingStatus.PARTIAL
                    if len(warnings) > warning_start
                    else ProcessingStatus.SUCCEEDED,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    input_count=input_count,
                    output_count=output_count(value),
                    duration_ms=int((perf_counter() - started) * 1000),
                    details={
                        **details,
                        **output_snapshot(value),
                        "warnings": warnings[warning_start:],
                    },
                )
            except Exception as exc:
                event = StageEvent(
                    stage=name,
                    status=ProcessingStatus.FAILED,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    input_count=input_count,
                    duration_ms=int((perf_counter() - started) * 1000),
                    message=str(exc),
                    error_code=type(exc).__name__,
                    details={**details, "warnings": warnings[warning_start:]},
                )
                await emit(event)
                raise
            await emit(event)
            return value

        try:
            plan = await stage(
                PipelineStage.QUERY_PLANNED,
                lambda: self.deps.query_planner.plan(
                    request.idea, request.candidate_limit_per_source
                ),
                component=self.deps.query_planner,
            )
            warnings.extend(plan.warnings)

            retrieval_slots = asyncio.Semaphore(self.max_concurrency)

            async def retrieve_one(retriever: PaperRetriever) -> list[Paper]:
                started_at = datetime.now(UTC)
                started = perf_counter()
                budget = RequestBudget(monotonic() + self.source_timeout_seconds)
                details = {
                    "queries": [
                        q.model_dump(mode="json")
                        for q in plan.queries
                        if q.source.value == retriever.source_name
                    ],
                    "demo": type(retriever).__name__.startswith("Demo"),
                }
                await emit(
                    StageEvent(
                        stage=PipelineStage.RETRIEVED,
                        scope="source",
                        source_name=retriever.source_name,
                        status=ProcessingStatus.RUNNING,
                        started_at=started_at,
                        details=details,
                    )
                )
                queries = [q for q in plan.queries if q.source.value == retriever.source_name]
                papers = []
                seen = set()
                attempts = []
                errors = []
                for query in queries or [None]:
                    query_start = perf_counter()
                    attempt_info = {
                        "query": query.query if query else "default",
                        "language": query.language if query else None,
                        "time_window": query.time_window if query else None,
                        "date_from": query.filters.get("date_from") if query else None,
                        "date_to": query.filters.get("date_to") if query else None,
                        "sort_by": query.sort_by if query else None,
                    }
                    if errors:
                        attempts.append(
                            {
                                **attempt_info,
                                "status": "skipped",
                                "count": 0,
                                "duration_ms": 0,
                                "message": "此来源本轮暂停，避免继续等待或重复请求。",
                            }
                        )
                        continue
                    try:
                        query_plan = plan.model_copy(update={"queries": [query]}) if query else plan
                        token = active_budget.set(budget)
                        try:
                            async with asyncio.timeout(budget.remaining()):
                                async with retrieval_slots:
                                    found = await retriever.search(query_plan)
                        finally:
                            active_budget.reset(token)
                        for paper in found:
                            identity = paper.doi or paper.arxiv_id or paper.title.casefold()
                            if identity not in seen:
                                papers.append(paper)
                                seen.add(identity)
                        attempts.append(
                            {
                                "query": query.query if query else "default",
                                "language": query.language if query else None,
                                "time_window": query.time_window if query else None,
                                "date_from": query.filters.get("date_from") if query else None,
                                "date_to": query.filters.get("date_to") if query else None,
                                "sort_by": query.sort_by if query else None,
                                "status": "succeeded",
                                "count": len(found),
                                "duration_ms": int((perf_counter() - query_start) * 1000),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - per-query provider boundary
                        errors.append(exc)
                        attempts.append(
                            {
                                "query": query.query if query else "default",
                                "language": query.language if query else None,
                                "time_window": query.time_window if query else None,
                                "date_from": query.filters.get("date_from") if query else None,
                                "date_to": query.filters.get("date_to") if query else None,
                                "sort_by": query.sort_by if query else None,
                                "status": "failed",
                                "count": 0,
                                "error": str(exc),
                                "error_code": type(exc).__name__,
                                "message": provider_message(exc, retriever.source_name),
                                "duration_ms": int((perf_counter() - query_start) * 1000),
                            }
                        )
                    await emit(
                        StageEvent(
                            stage=PipelineStage.RETRIEVED,
                            scope="source",
                            source_name=retriever.source_name,
                            status=ProcessingStatus.RUNNING,
                            started_at=started_at,
                            output_count=len(papers),
                            details={
                                **details,
                                "query_results": list(attempts),
                                **output_snapshot(papers),
                            },
                        )
                    )
                budget.cancelled.set()
                if errors:
                    warnings.append(provider_message(errors[-1], retriever.source_name))
                succeeded = any(a["status"] == "succeeded" for a in attempts)
                status = ProcessingStatus.PARTIAL if errors else ProcessingStatus.SUCCEEDED
                if not succeeded:
                    status = ProcessingStatus.FAILED
                if plan.source_limit:
                    papers = reserve_recent(
                        papers, plan.source_limit, plan.recent_from, plan.recent_share
                    )
                if plan.recent_from:
                    details["recency"] = {
                        "recent_from": plan.recent_from.isoformat(),
                        "recent_count": sum(is_recent(p, plan.recent_from) for p in papers),
                        "total": len(papers),
                        "target_share": plan.recent_share,
                    }
                await emit(
                    StageEvent(
                        stage=PipelineStage.RETRIEVED,
                        scope="source",
                        source_name=retriever.source_name,
                        status=status,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        output_count=len(papers),
                        duration_ms=int((perf_counter() - started) * 1000),
                        message=provider_message(errors[-1], retriever.source_name)
                        if errors
                        else None,
                        error_code=type(errors[-1]).__name__ if errors else None,
                        details={**details, "query_results": attempts, **output_snapshot(papers)},
                    )
                )
                return papers

            async def retrieve_all() -> list[Paper]:
                results = await asyncio.gather(
                    *(retrieve_one(retriever) for retriever in self.deps.retrievers),
                )
                papers = [paper for result in results for paper in result]
                if not papers:
                    raise RuntimeError(
                        "本轮未取得文献：部分来源可能暂不可用，或检索条件未匹配。请查看各来源状态。"
                    )
                return papers

            retrieved = await stage(PipelineStage.RETRIEVED, retrieve_all)
            deduped = await stage(
                PipelineStage.DEDUPED,
                lambda: self.deps.merger.merge_and_deduplicate(retrieved),
                len(retrieved),
                component=self.deps.merger,
            )
            metadata_papers[:] = deduped
            ranking_idea = request.idea.model_copy(
                update={
                    "search_terms": list(dict.fromkeys([*plan.concepts, *plan.synonyms])),
                    "date_from": plan.date_from or request.idea.date_from,
                    "date_to": plan.date_to or request.idea.date_to,
                    "recent_from": plan.recent_from or request.idea.recent_from,
                    "recent_share": plan.recent_share,
                }
            )
            pool = await stage(
                PipelineStage.PRE_RANKED,
                lambda: self.deps.pre_ranker.rank(
                    ranking_idea, deduped, request.pre_rank_pool_size
                ),
                len(deduped),
                component=self.deps.pre_ranker,
            )

            semaphore = asyncio.Semaphore(self.max_concurrency)
            demo_component = type(self.deps.fulltext_processor).__name__.startswith("Demo")

            def paper_details(paper: Paper) -> dict:
                return {"title": paper.title, "demo": demo_component}

            async def mark_running(
                stage_name: PipelineStage, paper: Paper
            ) -> tuple[datetime, float]:
                started_at = datetime.now(UTC)
                started = perf_counter()
                await emit(
                    StageEvent(
                        stage=stage_name,
                        scope="paper",
                        paper_id=paper.paper_id,
                        status=ProcessingStatus.RUNNING,
                        started_at=started_at,
                        details=paper_details(paper),
                    )
                )
                return started_at, started

            async def emit_outcome(
                stage_name: PipelineStage,
                paper: Paper,
                started_at: datetime,
                started: float,
                *,
                status: ProcessingStatus,
                message: str | None = None,
                output_count: int | None = None,
            ) -> None:
                await emit(
                    StageEvent(
                        stage=stage_name,
                        scope="paper",
                        paper_id=paper.paper_id,
                        status=status,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        output_count=output_count,
                        duration_ms=int((perf_counter() - started) * 1000),
                        message=message,
                        details={**paper_details(paper), "papers": [paper_snapshot(paper)]},
                    )
                )

            download_failures: dict[object, Exception] = {}

            async def settle_conversion(
                paper: Paper,
                started_at: datetime,
                started: float,
                converted: Paper | None,
                failure: Exception | None,
            ) -> Paper | None:
                outcome = ProcessingStatus.SUCCEEDED
                message = None
                if failure is not None:
                    # Adapter implementations come from independent groups and may
                    # wrap several third-party SDKs. This per-item boundary must
                    # convert any adapter failure into fallback/exclusion state.
                    if request.allow_abstract_fallback:
                        warnings.append(
                            f"{paper.title}: full text unavailable; abstract fallback ({failure})"
                        )
                        converted = paper
                        outcome = ProcessingStatus.PARTIAL
                        message = (
                            f"全文处理失败，保留{'摘要' if paper.abstract else '元数据'}：{failure}"
                        )
                        paper.fulltext.parse_error = paper.fulltext.parse_error or str(failure)
                    else:
                        warnings.append(
                            f"{paper.title}: excluded because full text failed ({failure})"
                        )
                        outcome = ProcessingStatus.SKIPPED
                        message = f"全文处理失败，排除文献：{failure}"
                if (
                    converted is not None
                    and request.require_fulltext
                    and not converted.fulltext.markdown_path
                ):
                    converted = None
                    outcome = ProcessingStatus.SKIPPED
                    message = f"仅保留全文已解析文献，排除当前文献。{message or ''}"
                if converted is None and failure is None:
                    outcome = ProcessingStatus.SKIPPED
                    message = "未取得公开全文，未进入 Markdown 转换。"
                await emit_outcome(
                    PipelineStage.MARKDOWN_CONVERTED,
                    paper,
                    started_at,
                    started,
                    status=outcome,
                    message=message,
                    output_count=int(converted is not None),
                )
                return converted

            async def download_one(paper: Paper) -> Paper | None:
                started_at, started = await mark_running(PipelineStage.DOWNLOADED, paper)
                failure = None
                try:
                    download = getattr(self.deps.fulltext_processor, "download", None)
                    if callable(download):
                        await download(paper)
                    else:
                        # 单步处理器（如联调组件）没有分段接口，整体放在下载阶段完成。
                        async with semaphore:
                            await self.deps.fulltext_processor.process(paper)
                except Exception as exc:  # noqa: BLE001 - 单篇失败必须降级而不是中断整段
                    failure = exc
                if failure is not None:
                    download_failures[paper.paper_id] = failure
                    await emit_outcome(
                        PipelineStage.DOWNLOADED,
                        paper,
                        started_at,
                        started,
                        status=ProcessingStatus.PARTIAL,
                        message=f"未取得公开全文：{failure}",
                        output_count=0,
                    )
                    return None
                await emit_outcome(
                    PipelineStage.DOWNLOADED,
                    paper,
                    started_at,
                    started,
                    status=ProcessingStatus.SUCCEEDED,
                    output_count=1,
                )
                return paper

            async def download_pool() -> list[Paper]:
                downloaded = await asyncio.gather(*(download_one(p) for p in pool))
                return [p for p in downloaded if p is not None]

            async def convert_one(paper: Paper) -> Paper | None:
                started_at, started = await mark_running(PipelineStage.MARKDOWN_CONVERTED, paper)
                converted = None
                failure = download_failures.get(paper.paper_id)
                if failure is None:
                    try:
                        convert = getattr(self.deps.fulltext_processor, "convert", None)
                        converted = await convert(paper) if callable(convert) else paper
                    except Exception as exc:  # noqa: BLE001 - 单篇失败必须降级而不是中断整段
                        failure = exc
                return await settle_conversion(paper, started_at, started, converted, failure)

            async def convert_pool(downloaded: list[Paper]) -> list[Paper]:
                # 未取得 PDF 的文献同样进入本阶段，用于展示“跳过转换”的原因。
                ready = {p.paper_id for p in downloaded}
                outcomes = await asyncio.gather(
                    *(
                        convert_one(paper)
                        if paper.paper_id in ready
                        else settle_conversion_without_download(paper)
                        for paper in pool
                    )
                )
                return [p for p in outcomes if p is not None]

            async def settle_conversion_without_download(paper: Paper) -> Paper | None:
                started_at, started = await mark_running(PipelineStage.MARKDOWN_CONVERTED, paper)
                return await settle_conversion(
                    paper, started_at, started, None, download_failures.get(paper.paper_id)
                )

            downloaded = await stage(
                PipelineStage.DOWNLOADED,
                download_pool,
                len(pool),
                component=self.deps.fulltext_processor,
            )
            processed = await stage(
                PipelineStage.MARKDOWN_CONVERTED,
                lambda: convert_pool(downloaded),
                len(pool),
                component=self.deps.fulltext_processor,
            )
            processed_by_id = {p.paper_id: p for p in pool}
            metadata_papers[:] = [processed_by_id.get(p.paper_id, p) for p in metadata_papers]
            if len(processed) < request.top_k:
                warnings.append(
                    f"Only {len(processed)} eligible papers remain for requested top_k={request.top_k}"
                )

            async def trace_paper(name, paper, operation):
                started_at = datetime.now(UTC)
                started = perf_counter()
                await emit(
                    StageEvent(
                        stage=name,
                        scope="paper",
                        paper_id=paper.paper_id,
                        status=ProcessingStatus.RUNNING,
                        started_at=started_at,
                        details={"title": paper.title},
                    )
                )
                try:
                    value = await operation()
                except Exception as exc:
                    await emit(
                        StageEvent(
                            stage=name,
                            scope="paper",
                            paper_id=paper.paper_id,
                            status=ProcessingStatus.FAILED,
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            duration_ms=int((perf_counter() - started) * 1000),
                            message=str(exc),
                            error_code=type(exc).__name__,
                            details={"title": paper.title},
                        )
                    )
                    raise
                await emit(
                    StageEvent(
                        stage=name,
                        scope="paper",
                        paper_id=paper.paper_id,
                        status=ProcessingStatus.SUCCEEDED,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        output_count=1,
                        duration_ms=int((perf_counter() - started) * 1000),
                        details={"title": paper.title, **output_snapshot(value)},
                    )
                )
                return value

            async def verify_all() -> list[Paper]:
                results = await asyncio.gather(
                    *(
                        trace_paper(
                            PipelineStage.VERIFIED,
                            p,
                            lambda p=p: self.deps.verifier.verify(ranking_idea, p),
                        )
                        for p in processed
                    ),
                    return_exceptions=True,
                )
                for value in results:
                    if isinstance(value, BaseException):
                        raise value
                return list(results)

            verified = await stage(
                PipelineStage.VERIFIED, verify_all, len(processed), component=self.deps.verifier
            )
            final_papers = await stage(
                PipelineStage.FINAL_RANKED,
                lambda: self.deps.final_ranker.rank(ranking_idea, verified, request.top_k),
                len(verified),
                component=self.deps.final_ranker,
            )

            async def summarize_one(paper: Paper):
                # 下载和总结处于不同阶段，但必须遵守同一批次的并发预算。
                async with semaphore:
                    return await trace_paper(
                        PipelineStage.SUMMARIZED,
                        paper,
                        lambda: self.deps.summarizer.summarize(request.idea, paper),
                    )

            async def summarize_all():
                results = await asyncio.gather(
                    *(summarize_one(p) for p in final_papers),
                    return_exceptions=True,
                )
                for value in results:
                    if isinstance(value, BaseException):
                        raise value
                return list(results)

            summaries = await stage(
                PipelineStage.SUMMARIZED,
                summarize_all,
                len(final_papers),
                component=self.deps.summarizer,
            )
            report = await stage(
                PipelineStage.REPORTED,
                lambda: self.deps.report_writer.write(request.idea, final_papers, summaries),
                len(final_papers),
                component=self.deps.report_writer,
            )
            status = ProcessingStatus.PARTIAL if warnings else ProcessingStatus.SUCCEEDED
            return PipelineResult(
                status=status,
                stage=PipelineStage.REPORTED,
                request=request,
                papers=final_papers,
                metadata_papers=metadata_papers,
                summaries=summaries,
                report=report,
                events=events,
                progress_events=progress_events,
                warnings=warnings,
            )
        # Top-level application boundary: MCP/UI callers receive a structured
        # failed result with audit events instead of an untyped transport crash.
        except Exception as exc:  # noqa: BLE001
            warnings.append(str(exc))
            return PipelineResult(
                status=ProcessingStatus.FAILED,
                stage=PipelineStage.FAILED,
                request=request,
                metadata_papers=metadata_papers,
                events=events,
                progress_events=progress_events,
                warnings=warnings,
            )
