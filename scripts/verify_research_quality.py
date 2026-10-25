"""Live research evaluation; stores stage outputs and full-text attempts for inspection."""

import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import PipelineRequest, ResearchIdea


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="场景文本检测与识别")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable model calls for a free reproducible check"
    )
    args = parser.parse_args()
    load_local_env()
    if args.no_llm:
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["PAPER_AGENT_LLM_API_KEY"] = ""
    settings = replace(
        Settings.from_env(),
        app_env="production",
        request_timeout_seconds=15,
        download_timeout_seconds=20,
        download_retry_limit=0,
    )
    agent = build_agent(settings)

    async def progress(event):
        if event.scope == "stage" or (event.scope == "source" and event.status.value != "running"):
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
            top_k=3,
            candidate_limit_per_source=6,
            pre_rank_pool_size=5,
        ),
        event_sink=progress,
    )
    destination = Path(settings.data_dir) / "quality-evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.run_id}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print("RESULT_FILE:", path.resolve(), flush=True)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "metadata_kept": len(result.metadata_papers),
                "selected": [
                    {
                        "title": p.title,
                        "score": p.scores.final,
                        "fulltext": bool(p.fulltext.markdown_path),
                        "reason": p.fulltext.parse_error,
                    }
                    for p in result.papers
                ],
                "review_length": len("".join(result.report.executive_summary.split()))
                if result.report
                else 0,
                "warnings": result.warnings,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
