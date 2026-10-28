"""第三组独立处理用例：公开论文下载、受限并发总结和报告，不执行检索排序。"""

from __future__ import annotations

import asyncio
import multiprocessing
import re
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from paper_agent.adapters import ProcessingComponents
from paper_agent.adapters.arxiv_metadata import ARXIV_ID
from paper_agent.domain.models import (
    Paper,
    PaperSummary,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    ResearchIdea,
    StageEvent,
)

MAX_BATCH_PAPERS = 100
MAX_WORKBENCH_CONCURRENCY = 8

EXAMPLE_TITLES = {
    "1706.03762": "Attention Is All You Need",
    "1409.0473": "Neural Machine Translation by Jointly Learning to Align and Translate",
}


def papers_from_arxiv_ids(value: str) -> list[Paper]:
    """仅接受公开 arXiv 标识，不把用户输入作为任意下载地址。"""
    ids = list(dict.fromkeys(part for part in re.split(r"[\s,，;；]+", value.strip()) if part))
    if not 1 <= len(ids) <= MAX_BATCH_PAPERS:
        raise ValueError(f"请填写 1 至 {MAX_BATCH_PAPERS} 个 arXiv 编号，每行一个。")
    if any(not ARXIV_ID.fullmatch(identifier) for identifier in ids):
        raise ValueError("编号格式不正确，例如 1706.03762；请不要填写网址。")
    return [
        Paper(
            title=EXAMPLE_TITLES.get(identifier, f"arXiv:{identifier}（待补充题名）"),
            arxiv_id=identifier,
            landing_page_url=f"https://arxiv.org/abs/{identifier}",
        )
        for identifier in ids
    ]


@dataclass
class PaperProcessingRecord:
    paper_id: str
    title: str
    fulltext_seconds: float = 0.0
    summary_seconds: float = 0.0
    fulltext_ok: bool = False
    summary_ok: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class ProcessingRun:
    result: PipelineResult
    records: list[PaperProcessingRecord]
    concurrency: int
    elapsed_seconds: float


class ProcessingWorkbench:
    """与九阶段主流程并列的第三组用例，仍通过相同 Protocol 调用适配器。"""

    def __init__(
        self,
        components_factory: Callable[[Executor], ProcessingComponents],
        *,
        metadata_enricher: Callable[[list[Paper]], Awaitable[list[str]]] | None = None,
    ):
        self._components_factory = components_factory
        self._metadata_enricher = metadata_enricher

    async def run(
        self,
        idea: ResearchIdea,
        papers: Sequence[Paper],
        *,
        concurrency: int = 4,
        progress: Callable[[str], None] | None = None,
    ) -> ProcessingRun:
        if not 1 <= len(papers) <= MAX_BATCH_PAPERS:
            raise ValueError(f"每批需提供 1 至 {MAX_BATCH_PAPERS} 篇论文。")
        if not 1 <= concurrency <= MAX_WORKBENCH_CONCURRENCY:
            raise ValueError(f"并发数必须在 1 至 {MAX_WORKBENCH_CONCURRENCY} 之间。")
        if len({paper.paper_id for paper in papers}) != len(papers):
            raise ValueError("论文 ID 不能重复。")
        started = perf_counter()
        copies = [paper.model_copy(deep=True) for paper in papers]
        records = [PaperProcessingRecord(str(p.paper_id), p.title) for p in copies]
        events: list[StageEvent] = []
        warnings: list[str] = []
        semaphore = asyncio.Semaphore(concurrency)
        notify = progress or (lambda message: None)
        if self._metadata_enricher is not None:
            notify("正在获取 arXiv 题名、作者、摘要等基本信息……")
            try:
                metadata_warnings = await self._metadata_enricher(copies)
                warnings.extend(metadata_warnings)
                notify(
                    "基本信息获取结束"
                    + ("，部分信息不可用，继续全文处理。" if metadata_warnings else "。")
                )
            except Exception as exc:  # 元数据异常不阻断已有全文处理链路。 # noqa: BLE001
                warnings.append(f"基本信息获取失败：{type(exc).__name__}；继续处理全文。")
                notify("基本信息获取失败，继续全文处理。")
            for paper, record in zip(copies, records, strict=True):
                record.title = paper.title

        counts = {
            stage: {state: 0 for state in ("成功", "失败", "跳过")}
            for stage in ("全文处理", "单篇总结")
        }

        def finished(stage: str, index: int, state: str) -> None:
            # 回调在父进程事件循环执行，累计数与输入序号分开，失败/跳过也计入已处理。
            count = counts[stage]
            count[state] += 1
            notify(
                f"{stage}：已处理 {sum(count.values())}/{len(copies)} 篇"
                f"（成功 {count['成功']}，失败 {count['失败']}，跳过 {count['跳过']}）；"
                f"第 {index + 1} 篇：{copies[index].title} · {state}"
            )

        def event(stage, began, clock, input_count, output_count):
            events.append(
                StageEvent(
                    stage=stage,
                    status=(
                        ProcessingStatus.SUCCEEDED
                        if input_count == output_count
                        else ProcessingStatus.PARTIAL
                        if output_count
                        else ProcessingStatus.FAILED
                    ),
                    started_at=began,
                    finished_at=datetime.now(UTC),
                    duration_ms=int((perf_counter() - clock) * 1000),
                    input_count=input_count,
                    output_count=output_count,
                )
            )

        # 文本层抽取使用多进程；远程下载、OCR 和 LLM 调用受信号量限制。
        with ProcessPoolExecutor(
            max_workers=concurrency, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            components = self._components_factory(executor)

            async def process_one(index: int):
                paper, record = copies[index], records[index]
                async with semaphore:
                    clock = perf_counter()
                    try:
                        copies[index] = await components.fulltext_processor.process(paper)
                        path = copies[index].fulltext.markdown_path
                        record.fulltext_ok = bool(path and Path(path).is_file())
                        if not record.fulltext_ok:
                            record.errors.append("未获得可用的全文文件。")
                    except Exception as exc:  # 独立论文失败不能中断整批任务。 # noqa: BLE001
                        record.errors.append(f"全文处理失败：{type(exc).__name__}")
                    record.fulltext_seconds = perf_counter() - clock
                finished("全文处理", index, "成功" if record.fulltext_ok else "失败")

            notify(f"开始全文处理：共 {len(copies)} 篇，最多 {concurrency} 篇并发。")
            began, clock = datetime.now(UTC), perf_counter()
            await asyncio.gather(*(process_one(i) for i in range(len(copies))))
            event(
                PipelineStage.FULLTEXT_PROCESSED,
                began,
                clock,
                len(copies),
                sum(record.fulltext_ok for record in records),
            )

            async def summarize_one(index: int) -> PaperSummary | None:
                paper, record = copies[index], records[index]
                if not record.fulltext_ok and not (paper.abstract or "").strip():
                    record.errors.append("没有可用全文或摘要，跳过总结。")
                    finished("单篇总结", index, "跳过")
                    return None
                async with semaphore:
                    clock = perf_counter()
                    try:
                        summary = await components.summarizer.summarize(idea, paper)
                        record.summary_ok = True
                        return summary
                    except Exception as exc:  # 保留其他论文已完成的总结。 # noqa: BLE001
                        record.errors.append(f"总结失败：{type(exc).__name__}")
                        return None
                    finally:
                        record.summary_seconds = perf_counter() - clock
                        finished("单篇总结", index, "成功" if record.summary_ok else "失败")

            notify(f"开始单篇总结：共 {len(copies)} 篇，最多 {concurrency} 篇并发。")
            began, clock = datetime.now(UTC), perf_counter()
            items = await asyncio.gather(*(summarize_one(i) for i in range(len(copies))))
            summaries = [item for item in items if item is not None]
            event(PipelineStage.SUMMARIZED, began, clock, len(copies), len(summaries))
            summary_ids = {summary.paper_id for summary in summaries}
            report_papers = [paper for paper in copies if paper.paper_id in summary_ids]
            report = None
            if summaries:
                began, clock = datetime.now(UTC), perf_counter()
                try:
                    notify("正在综合已成功生成的单篇总结……")
                    report = await components.report_writer.write(idea, report_papers, summaries)
                except Exception as exc:  # 报告失败不丢弃全文与单篇总结。 # noqa: BLE001
                    warnings.append(f"综合报告生成失败：{type(exc).__name__}")
                event(PipelineStage.REPORTED, began, clock, 1, int(report is not None))
        for record in records:
            warnings.extend(f"{record.title}：{error}" for error in record.errors)
        status = ProcessingStatus.SUCCEEDED if not warnings else ProcessingStatus.PARTIAL
        if not summaries:
            status = ProcessingStatus.FAILED
        result = PipelineResult(
            status=status,
            stage=(
                PipelineStage.FAILED
                if not summaries
                else PipelineStage.REPORTED
                if report
                else PipelineStage.SUMMARIZED
            ),
            request=PipelineRequest(idea=idea, top_k=len(copies), pre_rank_pool_size=len(copies)),
            papers=copies,
            summaries=summaries,
            report=report,
            events=events,
            warnings=warnings,
        )
        return ProcessingRun(result, records, concurrency, perf_counter() - started)
