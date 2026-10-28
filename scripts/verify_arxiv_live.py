"""Opt-in live API check; keeps real HTTP evidence, never uses fixture fallback.

Run from the project root after installation, or set PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree

from paper_agent.adapters import ArxivRetriever, ArxivSearchConfig, PaperMerger
from paper_agent.domain.models import Paper, PaperSource, SearchPlan, SearchQuery


async def verify(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "artifacts" / "arxiv_live" / stamp
    destination.mkdir(parents=True)
    config = ArxivSearchConfig(max_retries=args.retries)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "live_network_no_fixtures",
        "arxiv_sdk": version("arxiv"),
        "config": asdict(config),
        "implementation_sha256": hashlib.sha256(
            (root / "src/paper_agent/adapters/arxiv_retriever.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    retriever = ArxivRetriever(config=config)
    session = retriever.client._session
    exchanges = []
    attempts = []
    original_send = session.send

    def observe_send(request, **kwargs):
        started = time.monotonic()
        attempt = {"started_at": datetime.now(UTC).isoformat(), "url": request.url}
        try:
            response = original_send(request, **kwargs)
            attempt["status"] = response.status_code
            return response
        except Exception as exc:
            attempt.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            attempt["elapsed_seconds_including_rate_wait"] = round(time.monotonic() - started, 3)
            attempts.append(attempt)

    session.send = observe_send

    def capture(response, *unused, **kwargs):
        index = len(list(destination.glob("response-*.xml"))) + 1
        filename = f"response-{index:03d}.xml"
        body = response.content
        (destination / filename).write_bytes(body)
        exchanges.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "url": response.url,
                "status": response.status_code,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "body_file": filename,
                "headers": {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    in {
                        "content-type",
                        "date",
                        "age",
                        "x-cache",
                        "retry-after",
                        "server",
                        "via",
                    }
                },
            }
        )

    session.hooks["response"].append(capture)
    try:
        for query in args.query or ["electron"]:
            exchanges.clear()
            attempts.clear()
            started = time.monotonic()
            case = {
                "query": query,
                "limit": args.limit,
                "started_at": datetime.now(UTC).isoformat(),
            }
            try:
                plan = SearchPlan(
                    intent_summary="Live arXiv metadata verification",
                    queries=[
                        SearchQuery(source=PaperSource.ARXIV, query=query, limit=args.limit),
                    ],
                )
                papers = await retriever.search(plan)
                if not papers or len(papers) > args.limit:
                    raise ValueError("API must return 1..limit papers for this smoke query")
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = {}
                for exchange in exchanges:
                    if exchange["status"] == 200:
                        feed = ElementTree.parse(destination / exchange["body_file"])
                        for entry in feed.findall("atom:entry", ns):
                            entries[entry.findtext("atom:id", namespaces=ns)] = entry
                for paper in papers:
                    Paper.model_validate_json(paper.model_dump_json())
                    record = paper.source_records[0]
                    entry = entries.get(str(record.source_url))
                    if entry is None:
                        raise ValueError("Paper must trace to a real Atom entry")
                    expected_title = " ".join(entry.findtext("atom:title", "", ns).split())
                    expected_abstract = " ".join(entry.findtext("atom:summary", "", ns).split())
                    if paper.title != expected_title or paper.abstract != expected_abstract:
                        raise ValueError("Canonical title/abstract differs from API response")
                    if not (
                        paper.authors
                        and paper.publication_date
                        and paper.fulltext.pdf_url
                        and paper.arxiv_id
                        and paper.bibtex
                        and record.raw_metadata
                    ):
                        raise ValueError("Required smoke-check metadata is missing")
                    from paper_agent.adapters import BibTexMapper

                    exported = await BibTexMapper().to_paper(paper.bibtex, source="arxiv")
                    if (
                        exported.title != paper.title
                        or exported.year != paper.year
                        or exported.arxiv_id != paper.arxiv_id
                        or exported.doi != paper.doi
                        or len(exported.authors) != len(paper.authors)
                    ):
                        raise ValueError("Standalone BibTeX export lost available citation fields")
                merged = await PaperMerger().merge_and_deduplicate(papers + papers)
                if len(merged) != len(papers):
                    raise ValueError("Repeated live records were not deduplicated correctly")
                case.update(
                    status="passed",
                    count=len(papers),
                    papers=[paper.model_dump(mode="json") for paper in papers],
                    deduplicated_count=len(merged),
                )
                print(f"PASS {query!r}: {len(papers)} papers", flush=True)
                for paper in papers:
                    print(f"  {paper.arxiv_id}: {paper.title}", flush=True)
            except Exception as exc:  # noqa: BLE001 - persist every live-check failure as evidence
                case.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                print(f"FAIL {query!r}: {type(exc).__name__}: {exc}", flush=True)
            case["http"] = list(exchanges)
            case["transport_attempts"] = list(attempts)
            case["elapsed_seconds"] = round(time.monotonic() - started, 3)
            case["finished_at"] = datetime.now(UTC).isoformat()
            report["cases"].append(case)
    finally:
        session.close()
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["all_passed"] = bool(report["cases"]) and all(
            case["status"] == "passed" for case in report["cases"]
        )
        report_file = destination / "verification.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evidence: {report_file}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Repeat for multiple keyword queries")
    parser.add_argument("--limit", type=int, default=1, choices=range(1, 201), metavar="1..200")
    parser.add_argument("--retries", type=int, default=0, choices=range(4))
    raise SystemExit(asyncio.run(verify(parser.parse_args())))
