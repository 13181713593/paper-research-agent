"""显式运行真实完整流程或同池全文复测并导出记录；会消耗已配置 API 的额度。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from paper_agent.application.progress import paper_snapshot, redact_audit_value
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import (
    FullTextAsset,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    ResearchIdea,
    StageEvent,
)

ROOT = Path(__file__).resolve().parents[1]
redact = redact_audit_value


def save_json(path, data):
    path.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # 防止外部网页标题和错误信息被电子表格当作公式执行。
            clean = redact(row)
            writer.writerow(
                {
                    key: "'" + value
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                    else value
                    for key, value in clean.items()
                }
            )


def fulltext_events(result):
    """优先使用分段后的转换阶段快照，兼容仅有全文处理阶段的旧运行记录。"""
    for stage in (PipelineStage.MARKDOWN_CONVERTED, PipelineStage.FULLTEXT_PROCESSED):
        events = [
            event
            for event in result.progress_events
            if event.stage == stage and event.scope == "paper"
        ]
        if events:
            return events
    return []


def outcomes(result, *, ranked=True):
    final_ids = {str(p.paper_id) for p in result.papers}
    papers = {
        str(p.paper_id): p.model_dump(mode="json")
        for p in [*result.metadata_papers, *result.papers]
    }
    records, attempts = [], []
    for event in fulltext_events(result):
        snapshots = event.details.get("papers", [])
        paper = snapshots[0] if snapshots else papers.get(str(event.paper_id), {})
        fulltext = paper.get("fulltext", {})
        common = {
            "run_id": str(result.run_id),
            "paper_id": str(event.paper_id),
            "文献题目": event.details.get("title", paper.get("title", "")),
            "DOI": paper.get("doi") or "",
            "arXiv ID": paper.get("arxiv_id") or "",
        }
        pdf = fulltext.get("local_pdf_path")
        markdown = fulltext.get("markdown_path")
        records.append(
            {
                **common,
                "状态": event.status.value,
                "说明": event.message or "",
                "耗时（秒）": (event.duration_ms or 0) / 1000,
                "PDF存在": bool(pdf and Path(pdf).is_file()),
                "Markdown存在": bool(markdown and Path(markdown).is_file()),
                "进入Top-k": str(event.paper_id) in final_ids if ranked else None,
                "PDF路径": pdf or "",
                "Markdown路径": markdown or "",
                "最终错误": fulltext.get("parse_error") or "",
            }
        )
        for index, attempt in enumerate(fulltext.get("attempts", []), 1):
            attempts.append({**common, "序号": index, **attempt})
    return records, attempts


def export_result(result, destination, *, mode="full_production_pipeline"):
    records, attempts = outcomes(result, ranked=mode == "full_production_pipeline")
    filename = (
        "pipeline_result.json"
        if mode == "full_production_pipeline"
        else "fulltext_retry_result.json"
    )
    save_json(destination / filename, result.model_dump(mode="json"))
    save_json(destination / "fulltext_attempts.json", attempts)
    save_csv(
        destination / "fulltext_attempts.csv",
        attempts,
        [
            "run_id",
            "paper_id",
            "文献题目",
            "DOI",
            "arXiv ID",
            "序号",
            "stage",
            "status",
            "url",
            "error",
            "reason",
            "http_status",
            "candidate_count",
            "error_type",
            "error_kind",
        ],
    )
    save_csv(
        destination / "paper_outcomes.csv",
        records,
        [
            "run_id",
            "paper_id",
            "文献题目",
            "DOI",
            "arXiv ID",
            "状态",
            "说明",
            "耗时（秒）",
            "PDF存在",
            "Markdown存在",
            "进入Top-k",
            "PDF路径",
            "Markdown路径",
            "最终错误",
        ],
    )
    stats = {
        "run_id": str(result.run_id),
        "status": result.status.value,
        "stage": result.stage.value,
        "mode": mode,
        "processed": len(records),
        "pdf_available": sum(row["PDF存在"] for row in records),
        "markdown_available": sum(row["Markdown存在"] for row in records),
        "final_papers": len(result.papers),
        "summaries": len(result.summaries),
        "report_available": result.report is not None,
        "fulltext_statuses": dict(Counter(row["状态"] for row in records)),
        "channel_statuses": dict(Counter(f"{row['stage']}:{row['status']}" for row in attempts)),
    }
    save_json(destination / "statistics.json", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"审计记录目录：{destination}", flush=True)


def retry_pool(baseline):
    """从完整模型取同池文献，用全文终态快照还原缓存，不用精简快照重建 Paper。"""
    papers = {p.paper_id: p for p in [*baseline.metadata_papers, *baseline.papers]}
    pool = []
    seen = set()
    for event in fulltext_events(baseline):
        if event.paper_id in seen:
            continue
        seen.add(event.paper_id)
        if event.paper_id not in papers:
            raise ValueError(f"基线缺少完整 Paper 模型：{event.paper_id}")
        paper = papers[event.paper_id].model_copy(deep=True)
        snapshots = event.details.get("papers", [])
        if snapshots:
            paper.fulltext = FullTextAsset.model_validate(snapshots[0]["fulltext"])
        paper.fulltext.attempts = []
        paper.fulltext.parse_error = None
        pool.append(paper)
    if not pool:
        raise ValueError("基线没有全文候选池事件，不能进行同池复测")
    return pool


async def retry_fulltext(baseline, processor, record, concurrency=4):
    pool = retry_pool(baseline)
    semaphore = asyncio.Semaphore(concurrency)
    events = []
    started_at = datetime.now(UTC)
    started = perf_counter()

    async def process(paper):
        async with semaphore:
            paper_start = perf_counter()
            paper_started_at = datetime.now(UTC)
            status, message, error_code = ProcessingStatus.SUCCEEDED, None, None
            had_markdown = bool(
                paper.fulltext.markdown_path and Path(paper.fulltext.markdown_path).is_file()
            )
            try:
                await processor.process(paper)
                if had_markdown and not paper.fulltext.attempts:
                    paper.fulltext.attempts.append(
                        {
                            "stage": "cache",
                            "status": "succeeded",
                            "reason": "复用有效 PDF/Markdown 缓存",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - 与主流程一致，单篇失败保留并继续
                status = ProcessingStatus.PARTIAL
                error_code = type(exc).__name__
                message = paper.fulltext.parse_error or f"{error_code}: {exc}"
            event = StageEvent(
                stage=PipelineStage.FULLTEXT_PROCESSED,
                scope="paper",
                paper_id=paper.paper_id,
                status=status,
                started_at=paper_started_at,
                finished_at=datetime.now(UTC),
                duration_ms=int((perf_counter() - paper_start) * 1000),
                output_count=1,
                message=message,
                error_code=error_code,
                details={"title": paper.title, "papers": [paper_snapshot(paper)]},
            )
            events.append(event)
            await record(event)

    await asyncio.gather(*(process(paper) for paper in pool))
    status = (
        ProcessingStatus.PARTIAL
        if any(e.status == ProcessingStatus.PARTIAL for e in events)
        else ProcessingStatus.SUCCEEDED
    )
    terminal = StageEvent(
        stage=PipelineStage.FULLTEXT_PROCESSED,
        status=status,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_ms=int((perf_counter() - started) * 1000),
        input_count=len(pool),
        output_count=len(pool),
        details={"mode": "same_pool_fulltext_retry", "papers": [paper_snapshot(p) for p in pool]},
    )
    events.append(terminal)
    await record(terminal)
    return PipelineResult(
        status=status,
        stage=PipelineStage.FULLTEXT_PROCESSED,
        request=baseline.request,
        metadata_papers=pool,
        events=[terminal],
        progress_events=events,
        warnings=["仅复测同池全文；未重新检索、排序、总结或生成报告。"],
    )


def export_comparison(baseline_path, after, destination):
    # 基线统计使用当时保存的布尔值，不能被复测后新增磁盘缓存影响。
    with baseline_path.with_name("paper_outcomes.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        before = {row["paper_id"]: row for row in csv.DictReader(handle)}
    after_rows, _ = outcomes(after, ranked=False)
    if set(before) != {row["paper_id"] for row in after_rows}:
        raise ValueError("前后候选池不相同，不能导出同池比较")
    rows = [
        {
            "paper_id": row["paper_id"],
            "文献题目": row["文献题目"],
            "DOI": row["DOI"],
            "基线PDF": before[row["paper_id"]]["PDF存在"] == "True",
            "复测PDF": row["PDF存在"],
            "基线Markdown": before[row["paper_id"]]["Markdown存在"] == "True",
            "复测Markdown": row["Markdown存在"],
            "复测说明": row["说明"],
        }
        for row in after_rows
    ]
    save_csv(destination / "comparison.csv", rows, list(rows[0]))
    save_json(
        destination / "comparison.json",
        {
            "mode": "same_pool_fulltext_retry",
            "baseline_path": str(baseline_path),
            "pool_size": len(rows),
            "before_pdf": sum(row["基线PDF"] for row in rows),
            "after_pdf": sum(row["复测PDF"] for row in rows),
            "before_markdown": sum(row["基线Markdown"] for row in rows),
            "after_markdown": sum(row["复测Markdown"] for row in rows),
            "papers": rows,
            "note": "同池网络复测包含服务状态波动；新增成功需结合逐渠道记录确认，不能全部归因于代码。",
        },
    )


async def run(args):
    load_local_env(ROOT / ".env")
    # 当前项目优先；复用虚拟环境所属项目仅补充缺失的总结配置。
    fallback = Path(sys.executable).parents[2] / ".env"
    if fallback != ROOT / ".env":
        load_local_env(fallback)
    settings = replace(
        Settings.from_env(),
        app_env="production",
        data_dir=str(ROOT / "data"),
        max_concurrency=4,
    )
    agent = build_agent(settings)
    deps = agent._pipeline.deps
    if any(
        type(component).__name__.startswith("Demo")
        for component in (
            deps.query_planner,
            deps.pre_ranker,
            deps.fulltext_processor,
            deps.verifier,
            deps.final_ranker,
            deps.summarizer,
            deps.report_writer,
            *deps.retrievers,
        )
    ):
        raise RuntimeError("实跑不能混入 Demo 组件")
    baseline = (
        PipelineResult.model_validate_json(args.retry_from.read_text("utf-8"))
        if args.retry_from
        else None
    )
    mode = "same_pool_fulltext_retry" if baseline else "full_production_pipeline"
    destination = (
        ROOT
        / "artifacts"
        / "fulltext_audit"
        / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + ("same_pool_after" if baseline else args.label)
        )
    )
    destination.mkdir(parents=True, exist_ok=False)
    save_json(
        destination / "runtime.json",
        {
            "mode": mode,
            "idea": baseline.request.idea.text if baseline else args.idea,
            "baseline_path": str(args.retry_from) if baseline else None,
            "top_k": None if baseline else args.top_k,
            "pool_size": len(retry_pool(baseline)) if baseline else args.pool_size,
            "limit_per_source": None if baseline else args.limit,
            "markdown_backend": settings.markdown_backend,
            "unpaywall_configured": bool(settings.unpaywall_email.strip()),
            "openalex_configured": bool(os.getenv("OPENALEX_API_KEY", "").strip()),
            "mineru_configured": bool(os.getenv("MINERU_API_KEY", "").strip()),
            "summary_api_configured": deps.summarizer._client is not None,
        },
    )
    print(f"真实运行模式：{mode}；记录目录：{destination}", flush=True)

    async def record(event):
        if event.status == ProcessingStatus.RUNNING:
            return
        with (destination / "progress_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(redact(event.model_dump(mode="json")), ensure_ascii=False) + "\n"
            )
        if event.scope in {"stage", "source"} or event.stage == PipelineStage.FULLTEXT_PROCESSED:
            print(
                redact(
                    f"{event.stage.value}/{event.scope}: {event.status.value}; "
                    f"{event.source_name or event.details.get('title', '')}; "
                    f"输出={event.output_count}; {event.message or ''}"
                ),
                flush=True,
            )

    if baseline:
        result = await retry_fulltext(baseline, deps.fulltext_processor, record)
        export_comparison(args.retry_from, result, destination)
    else:
        result = await agent.research(
            PipelineRequest(
                idea=ResearchIdea(text=args.idea),
                top_k=args.top_k,
                pre_rank_pool_size=args.pool_size,
                candidate_limit_per_source=args.limit,
                allow_abstract_fallback=True,
                require_fulltext=False,
            ),
            event_sink=record,
        )
    export_result(result, destination, mode=mode)
    return int(result.status == ProcessingStatus.FAILED)


async def probe_page(url):
    """保存公开页面的脱敏 HTML，验证下载链接识别，不发送 API 凭证。"""
    import httpx

    from paper_agent.adapters.open_access_resolver import _PdfLinks

    destination = ROOT / "artifacts" / "fulltext_audit" / "page_probes"
    destination.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, max_redirects=5) as client:  # noqa: SIM117
        async with client.stream("GET", url) as response:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 2_000_000:
                    raise ValueError("页面超过 2 MB，不继续诊断")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            target = destination / f"{stamp}.html"
            html = body.decode("utf-8", errors="replace")
            target.write_text(redact(html), encoding="utf-8")
            parser = _PdfLinks(str(response.url))
            parser.feed(html)
            info = {
                "url": url,
                "final_url": str(response.url),
                "http_status": response.status_code,
                "pdf_candidates": parser.urls,
                "html_path": str(target),
            }
            save_json(target.with_suffix(".json"), info)
            print(json.dumps(redact(info), ensure_ascii=False))


def inspect_events(path):
    """只读查看已有实跑日志，不再次调用 API。"""
    for line in Path(path).read_text("utf-8").splitlines():
        event = json.loads(line)
        if event["stage"] != "fulltext_processed" or event["scope"] != "paper":
            continue
        paper = event.get("details", {}).get("papers", [{}])[0]
        fulltext = paper.get("fulltext", {})
        print(
            json.dumps(
                redact(
                    {
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "status": event["status"],
                        "attempts": fulltext.get("attempts", []),
                        "parse_error": fulltext.get("parse_error"),
                    }
                ),
                ensure_ascii=False,
            )
        )


def verify_pdfs(path):
    """只读核对已下载 PDF 的第一页，排除期刊指南等错误附件；不调用付费解析。"""
    from pypdf import PdfReader

    result = PipelineResult.model_validate_json(path.read_text("utf-8"))
    identities = []
    for event in fulltext_events(result):
        paper = event.details["papers"][0]
        pdf = paper.get("fulltext", {}).get("local_pdf_path")
        if not pdf or not Path(pdf).is_file():
            continue
        reader = PdfReader(pdf)
        identities.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "doi": paper.get("doi"),
                "pages": len(reader.pages),
                "pdf_path": pdf,
                "first_page_text": (reader.pages[0].extract_text() or "")[:4000],
            }
        )
    target = path.with_name("pdf_identity_checks.json")
    save_json(target, identities)
    print(
        json.dumps(
            {
                "checked": len(identities),
                "identity_checks_path": str(target),
                "papers": [{"title": item["title"], "pages": item["pages"]} for item in identities],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="推荐系统 recommendation system")
    parser.add_argument("--label", default="baseline", choices=["baseline", "after"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=30)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--inspect", type=Path, help="只读查看 progress_events.jsonl")
    action.add_argument("--probe-url", help="验证公开落地页的 PDF 链接识别")
    action.add_argument(
        "--retry-from", type=Path, help="用完整基线 JSON 同池复测全文，复用缓存，不重跑总结"
    )
    action.add_argument("--verify-pdfs-from", type=Path, help="只读核对已下载 PDF 第一页身份")
    args = parser.parse_args()
    if args.inspect:
        inspect_events(args.inspect)
    elif args.probe_url:
        asyncio.run(probe_page(args.probe_url))
    elif args.verify_pdfs_from:
        verify_pdfs(args.verify_pdfs_from)
    else:
        raise SystemExit(asyncio.run(run(args)))
