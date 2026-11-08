"""Explicit live acceptance run; writes ignored local artifacts, never credentials.

Usage: python scripts/validate_research_quality.py --live --idea '研究问题'
This calls the configured model and public providers. A completed pipeline alone
does not pass: full text, model usage and source quotes are independently checked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from paper_agent.adapters.short_review import review_length
from paper_agent.adapters.source_evidence import has_substantive_description, quote_in_source
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import (
    PipelineRequest,
    PipelineResult,
    ProcessingStatus,
    ResearchIdea,
)


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k != "raw_metadata"}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def acceptance_checks(result, args):
    registry = {s.paper_id: s for s in result.summaries}
    checks = {
        "pipeline_reached_report": result.report is not None
        and result.status != ProcessingStatus.FAILED,
        "requested_paper_count": len(result.papers) == args.top_k,
        "abstracts_present": bool(result.papers) and all(p.abstract for p in result.papers),
        "fulltexts_present": bool(result.papers)
        and all(
            p.fulltext.markdown_path and Path(p.fulltext.markdown_path).is_file()
            for p in result.papers
        ),
        "all_summaries_used_model": len(result.summaries) == args.top_k
        and all(s.generation_method == "model" for s in result.summaries),
        "report_used_model": bool(result.report and result.report.generation_method == "model"),
        "report_length": bool(
            result.report and 1000 <= review_length(result.report.executive_summary) <= 1500
        ),
    }
    citations = []
    for paper in result.papers:
        summary = registry.get(paper.paper_id)
        source = (
            Path(paper.fulltext.markdown_path).read_text("utf-8")
            if paper.fulltext.markdown_path and Path(paper.fulltext.markdown_path).is_file()
            else (paper.abstract or "")
        )
        citations.append(
            bool(summary and summary.evidence)
            and all(ref.quote and quote_in_source(ref.quote, source) for ref in summary.evidence)
        )
    checks["all_quotes_match_source"] = len(citations) == args.top_k and all(citations)
    if args.require_resource_comparison:
        checks["resource_comparison_fields"] = len(result.summaries) == args.top_k and all(
            s.architecture.strip()
            and has_substantive_description(s.architecture)
            and s.resource_allocation.strip()
            and has_substantive_description(s.resource_allocation)
            and s.resource_constraints
            and s.evaluation_metrics
            for s in result.summaries
        )
    return checks


async def run(args):
    load_local_env()
    settings = Settings.from_env()
    if settings.app_env != "production":
        raise SystemExit("验收要求 PAPER_AGENT_ENV=production")
    if args.reaudit:
        result = PipelineResult.model_validate(json.loads(Path(args.reaudit).read_text("utf-8")))
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
    else:
        agent = build_agent(settings)
        deps = agent._pipeline.deps
        if not deps.query_planner.concept_extractor.active or deps.summarizer._client is None:
            raise SystemExit("验收要求本地配置可用的查询规划与总结模型 Key")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        async def progress(event):
            if event.status != ProcessingStatus.RUNNING and event.scope in {"stage", "source"}:
                print(
                    json.dumps(
                        {
                            "stage": event.stage.value,
                            "source": event.source_name,
                            "status": event.status.value,
                            "count": event.output_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        result = await agent.research(
            PipelineRequest(
                idea=ResearchIdea(text=args.idea),
                top_k=args.top_k,
                pre_rank_pool_size=args.pool_size,
                candidate_limit_per_source=args.source_limit,
                require_fulltext=True,
            ),
            event_sink=progress,
        )
    checks = acceptance_checks(result, args)
    data = scrub(result.model_dump(mode="json"))
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    for key in (
        "OPENAI_API_KEY",
        "PAPER_AGENT_LLM_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "OPENALEX_API_KEY",
    ):
        secret = os.getenv(key, "")
        if secret and secret in serialized:
            raise SystemExit("凭证保护检查失败，未写出结果")
    (output / "pipeline.json").write_text(serialized, "utf-8")
    if result.report:
        (output / "research_report.md").write_text(result.report.markdown, "utf-8")
    audit = {
        "checks": checks,
        "verification_mode": "reaudit" if args.reaudit else "live",
        "passed": all(checks.values()),
        "papers": [
            {
                "title": p.title,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "fulltext": bool(p.fulltext.markdown_path),
            }
            for p in result.papers
        ],
        "warnings": result.warnings,
        "limits": "引文文字匹配不验证论文结论、页码、实验复现或检索穷尽性；需人工复核。",
    }
    (output / "acceptance.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), "utf-8"
    )
    print(
        json.dumps(
            {"checks": checks, "passed": audit["passed"], "output": str(output)}, ensure_ascii=False
        ),
        flush=True,
    )
    return 0 if audit["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument(
        "--reaudit", help="Re-check a local exported pipeline.json without network or model calls"
    )
    parser.add_argument("--idea")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--source-limit", type=int, default=15)
    parser.add_argument("--require-resource-comparison", action="store_true")
    parser.add_argument(
        "--output-dir", default="artifacts/quality-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    if args.live and not args.idea:
        parser.error("--live requires --idea")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
