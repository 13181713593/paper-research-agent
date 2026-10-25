"""Verify the configured model against an existing real retrieval result."""

import argparse
import asyncio
import json
from pathlib import Path

from paper_agent.adapters import build_processing_components
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import PipelineResult


async def main():
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="JSON result from verify_research_quality.py")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Regenerate individual summaries with configured model",
    )
    args = parser.parse_args()
    folder = Path("data/quality-evaluation")
    result = PipelineResult.model_validate_json(args.result.read_text(encoding="utf-8"))
    components = build_processing_components(Settings.from_env())
    original = components.report_writer._client

    class ObservedClient:
        async def complete(self, messages, **kwargs):
            try:
                raw = await original.complete(messages, **kwargs)
            except Exception as exc:
                print("MODEL_ERROR:", type(exc).__name__, str(exc)[:180], flush=True)
                raise
            print("MODEL_RESPONSE_CHARS:", len(raw), flush=True)
            return raw

    if original is None:
        raise RuntimeError("No report model configured")
    if args.summarize:
        result.summaries = await asyncio.gather(
            *[components.summarizer.summarize(result.request.idea, p) for p in result.papers]
        )
        print("SUMMARIES_COMPLETED:", len(result.summaries), flush=True)
    components.report_writer._client = ObservedClient()
    result.report = await components.report_writer.write(
        result.request.idea, result.papers, result.summaries
    )
    print(result.report.markdown, flush=True)
    (folder / "model-review.md").write_text(result.report.markdown, encoding="utf-8")
    (folder / "verified-result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"model_success": "模型综合" in result.report.markdown}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
