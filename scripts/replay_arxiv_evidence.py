"""Replay a saved successful HTTP response; this is explicitly NOT a live check."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import arxiv
import feedparser

from paper_agent.adapters import ArxivRetriever, BibTexMapper, PaperMerger
from paper_agent.domain.models import Paper


async def replay(evidence: Path, output_dir: Path | None = None) -> int:
    source = json.loads(evidence.read_text(encoding="utf-8"))
    report = {"mode": "replay_saved_real_http_no_network", "at": datetime.now(UTC).isoformat(),
              "original_report": str(evidence.resolve()), "response_sha256": []}
    retriever = ArxivRetriever()
    destination = output_dir or (Path(__file__).resolve().parents[1] / "artifacts" / "final_review")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        papers = []
        for case in source["cases"]:
            for exchange in case["http"]:
                if exchange.get("status") != 200:
                    continue
                body = (evidence.parent / exchange["body_file"]).read_bytes()
                digest = hashlib.sha256(body).hexdigest()
                if digest != exchange["sha256"]:
                    raise ValueError("Saved HTTP response hash differs from original evidence")
                report["response_sha256"].append(digest)
                for entry in feedparser.parse(body).entries:
                    paper = await retriever._map_result(arxiv.Result._from_feed_entry(entry))
                    exported = await BibTexMapper().to_paper(paper.bibtex, source="arxiv")
                    for field in ("title", "year", "doi", "arxiv_id", "publication_date"):
                        if getattr(exported, field) != getattr(paper, field):
                            raise ValueError(f"Standalone BibTeX lost {field}")
                    if [a.name for a in exported.authors] != [a.name for a in paper.authors]:
                        raise ValueError("Standalone BibTeX changed author names")
                    if not paper.abstract or not paper.source_records[0].raw_metadata:
                        raise ValueError("Saved real result lost abstract/provenance")
                    Paper.model_validate_json(paper.model_dump_json())
                    papers.append(paper)
        if not papers:
            raise ValueError("No successful real arXiv entries in the specified evidence")
        unique = await PaperMerger().merge_and_deduplicate(papers)
        repeated = await PaperMerger().merge_and_deduplicate(papers + papers)
        if [p.model_dump() for p in unique] != [p.model_dump() for p in repeated]:
            raise ValueError("Replay deduplication is not idempotent")
        report.update(status="passed", count=len(papers),
                      papers=[p.model_dump(mode="json") for p in papers])
        (destination / "arxiv_export.bib").write_text(
            "\n\n".join(p.bibtex for p in unique) + "\n", encoding="utf-8")
        print(f"PASS saved-response replay: {len(papers)} Papers; NO live request", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics preserve failures as well
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        print(f"FAIL saved-response replay: {exc}", flush=True)
    finally:
        retriever.client._session.close()
        target = destination / "arxiv_saved_replay.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evidence: {target}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path,
                        help="A previous verify_arxiv_live.py verification.json")
    parser.add_argument("--output-dir", type=Path, help="Separate replay output directory")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(replay(args.evidence, args.output_dir)))
