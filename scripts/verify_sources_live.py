"""Opt-in public API diagnostics, not B's production retrievers.

Save actual responses, map real records to the shared Paper, and check A's merger.
No API keys, fixture fallback, PDF download, or bootstrap changes are used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import requests

from paper_agent.adapters import BibTexMapper, PaperMerger
from paper_agent.adapters.retrieval_common import normalize_doi, normalize_title
from paper_agent.domain.models import Author, FullTextAsset, Paper, PaperSource, SourceRecord

SOURCES = ("crossref", "openalex", "semantic_scholar")


class AbstractText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def restore_abstract(index):
    if not index:
        return None
    words = {}
    for token, positions in index.items():
        for position in positions:
            if not isinstance(position, int) or position < 0 or position in words:
                raise ValueError("Invalid or duplicate OpenAlex abstract position")
            words[position] = token
    if not words or sorted(words) != list(range(len(words))):
        raise ValueError("OpenAlex abstract has missing positions")
    return " ".join(words[position] for position in range(len(words)))


def openalex_paper(record):
    """Diagnostic JSON mapping; preserve the provider payload without inventing BibTeX."""
    authors = [
        Author(
            name=item["author"]["display_name"],
            orcid=item["author"].get("orcid"),
            affiliations=[
                institution["display_name"]
                for institution in item.get("institutions", [])
                if institution.get("display_name")
            ],
        )
        for item in record.get("authorships", [])
    ]
    location = record.get("primary_location") or {}
    oa_location = record.get("best_oa_location") or {}
    oa = record.get("open_access") or {}
    landing = location.get("landing_page_url") or record["id"]
    return Paper(
        paper_id=uuid5(NAMESPACE_URL, f"paper-agent:openalex:{record['id']}"),
        title=record["display_name"],
        normalized_title=normalize_title(record["display_name"]),
        authors=authors,
        institutions=list(
            dict.fromkeys(name for author in authors for name in author.affiliations)
        ),
        year=record.get("publication_year"),
        publication_date=record.get("publication_date"),
        doi=normalize_doi(record.get("doi")),
        openalex_id=record["id"],
        abstract=restore_abstract(record.get("abstract_inverted_index")),
        language=record.get("language"),
        citation_count=record.get("cited_by_count"),
        is_open_access=oa.get("is_oa"),
        is_retracted=record.get("is_retracted"),
        landing_page_url=landing,
        open_access_url=oa.get("oa_url"),
        fulltext=FullTextAsset(pdf_url=oa_location.get("pdf_url"), landing_page_url=landing),
        source_records=[
            SourceRecord(
                source=PaperSource.OPENALEX,
                source_id=record["id"],
                source_url=record["id"],
                raw_metadata=record,
            )
        ],
    )


def semantic_scholar_paper(record):
    external = record.get("externalIds") or {}
    return Paper(
        paper_id=uuid5(NAMESPACE_URL, f"paper-agent:semantic_scholar:{record['paperId']}"),
        title=record["title"],
        authors=[Author(name=a["name"]) for a in record.get("authors", [])],
        year=record.get("year"),
        abstract=record.get("abstract"),
        doi=normalize_doi(external.get("DOI")),
        arxiv_id=external.get("ArXiv"),
        semantic_scholar_id=record["paperId"],
        landing_page_url=record.get("url"),
        source_records=[
            SourceRecord(
                source=PaperSource.SEMANTIC_SCHOLAR,
                source_id=record["paperId"],
                source_url=record.get("url"),
                raw_metadata=record,
            )
        ],
    )


class LiveEvidence:
    def __init__(self, destination):
        self.destination = destination
        self.sequence = 0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "paper-research-agent/0.2 (academic course project)"}
        )

    def get(self, case, url, *, params=None, bibtex=False):
        self.sequence += 1
        filename = f"response-{self.sequence:03d}.{'bib' if bibtex else 'json'}"
        exchange = {"at": datetime.now(UTC).isoformat(), "requested_url": url, "params": params}
        case.setdefault("http", []).append(exchange)
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=(10, 25),
                headers={
                    "Accept": "application/x-bibtex" if bibtex else "application/json",
                },
            )
        except requests.RequestException as exc:
            exchange.update(error_type=type(exc).__name__, error=str(exc))
            raise
        body = response.content
        (self.destination / filename).write_bytes(body)
        exchange.update(
            url=response.url,
            status=response.status_code,
            body_file=filename,
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower()
                in {
                    "content-type",
                    "date",
                    "retry-after",
                    "x-ratelimit-limit",
                    "x-ratelimit-remaining",
                }
            },
        )
        response.raise_for_status()
        # Crossref transform may omit Content-Type; its payload is UTF-8.
        return body.decode("utf-8-sig") if bibtex else response.json()

    async def crossref_paper(self, case, record):
        doi = record["DOI"]
        url = (
            f"https://api.crossref.org/works/{quote(doi, safe='/')}/transform/application/x-bibtex"
        )
        bibtex = self.get(case, url, bibtex=True)
        supplemental = {
            "doi": doi,
            "source_id": doi,
            "source_url": record.get("URL"),
            "raw_metadata": record,
        }
        # Some real transform responses omit even the title; use the same DOI's JSON.
        if record.get("title"):
            supplemental["title"] = record["title"][0]
        if record.get("abstract"):
            parser = AbstractText()
            parser.feed(record["abstract"])
            supplemental["abstract"] = " ".join(" ".join(parser.parts).split()) or None
        if record.get("author"):
            supplemental["authors"] = [
                {
                    "name": a.get("name")
                    or " ".join(filter(None, [a.get("given"), a.get("family")])),
                    "orcid": a.get("ORCID"),
                    "affiliations": [i["name"] for i in a.get("affiliation", []) if i.get("name")],
                }
                for a in record["author"]
            ]
        for field in ("published", "issued"):
            parts = (record.get(field) or {}).get("date-parts") or []
            if parts and parts[0]:
                supplemental["year"] = parts[0][0]
                if len(parts[0]) == 3:
                    year, month, day = parts[0]
                    supplemental["publication_date"] = f"{year:04d}-{month:02d}-{day:02d}"
                break
        paper = await BibTexMapper().to_paper(
            bibtex, source="crossref", supplemental_metadata=supplemental
        )
        if paper.doi != normalize_doi(doi) or not paper.bibtex:
            raise ValueError("BibTeX mapping lost the DOI or original BibTeX")
        return paper


async def check_papers(case, papers):
    if not papers:
        raise ValueError("The smoke query returned no papers")
    for paper in papers:
        restored = Paper.model_validate_json(paper.model_dump_json())
        if restored != paper or not paper.source_records[0].raw_metadata:
            raise ValueError("Paper JSON round-trip or provenance check failed")
    merger = PaperMerger()
    unique = await merger.merge_and_deduplicate(papers)
    repeated = await merger.merge_and_deduplicate(papers + papers)
    if [p.model_dump() for p in repeated] != [p.model_dump() for p in unique]:
        raise ValueError("Repeating real records changed the deduplicated result")
    case.update(
        papers=[p.model_dump(mode="json") for p in papers],
        count=len(papers),
        abstract_count=sum(bool(p.abstract) for p in papers),
        deduplicated_count=len(unique),
        json_roundtrip="passed",
        repeat_merge="passed",
    )


def record_failure(case, exc):
    case.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    print(f"FAIL {case['source']} {case['query']!r}: {exc}", flush=True)


async def verify(args):
    destination = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "multisource_live"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    destination.mkdir(parents=True)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "live_network_no_fixtures",
        "authentication": "no_api_keys",
        "scope": "API and shared model diagnostics; production B adapters are not wired",
        "cases": [],
    }
    client = LiveEvidence(destination)
    openalex_candidates = []
    try:
        for query in args.query or ["neural retrieval"]:
            for source in args.source or SOURCES:
                case = {"source": source, "query": query, "limit": args.limit}
                report["cases"].append(case)
                try:
                    if source == "crossref":
                        payload = client.get(
                            case,
                            "https://api.crossref.org/works",
                            params={"query": query, "rows": args.limit},
                        )
                        records = payload["message"]["items"]
                    elif source == "openalex":
                        payload = client.get(
                            case,
                            "https://api.openalex.org/works",
                            params={"search": query, "per-page": args.limit},
                        )
                        records = payload["results"]
                    else:
                        payload = client.get(
                            case,
                            "https://api.semanticscholar.org/graph/v1/paper/search",
                            params={
                                "query": query,
                                "limit": args.limit,
                                "fields": "title,authors,year,abstract,url,externalIds",
                            },
                        )
                        records = payload["data"]
                    case.update(search_status="passed", returned_count=len(records))
                    if not 1 <= len(records) <= args.limit:
                        raise ValueError("Expected 1..limit real results for the smoke query")
                    if source == "crossref":
                        papers = [await client.crossref_paper(case, r) for r in records]
                    elif source == "openalex":
                        papers = [openalex_paper(r) for r in records]
                    else:
                        papers = [semantic_scholar_paper(r) for r in records]
                    await check_papers(case, papers)
                    case["status"] = "passed"
                    if source == "openalex":
                        openalex_candidates.extend(p for p in papers if p.doi)
                    print(f"PASS {source} {query!r}: {len(papers)} Papers", flush=True)
                    for paper in papers:
                        print(f"  {paper.doi or paper.openalex_id}: {paper.title}", flush=True)
                except Exception as exc:  # noqa: BLE001 - every failed live check is persisted
                    record_failure(case, exc)

        # Fetch the same DOI from Crossref to test a genuinely cross-provider merge.
        if openalex_candidates and "crossref" in (args.source or SOURCES):
            case = {
                "source": "openalex+crossref",
                "query": "first overlapping DOI",
                "kind": "same_doi_lookup_and_merge",
            }
            report["cases"].append(case)
            try:
                # OpenAlex covers DOIs from multiple registries, not only Crossref.
                # Record Crossref 404s as coverage gaps; never treat other errors as empty.
                for original in openalex_candidates[:3]:
                    try:
                        payload = client.get(
                            case, f"https://api.crossref.org/works/{quote(original.doi, safe='/')}"
                        )
                        break
                    except requests.HTTPError as exc:
                        if exc.response.status_code != 404:
                            raise
                        case.setdefault("not_found_in_crossref", []).append(original.doi)
                else:
                    raise ValueError(
                        "No overlapping Crossref DOI in the first three OpenAlex records"
                    )
                case["query"] = original.doi
                crossref = await client.crossref_paper(case, payload["message"])
                await check_papers(case, [original, crossref])
                merged = await PaperMerger().merge_and_deduplicate([original, crossref])
                if len(merged) != 1 or {s.source for s in merged[0].source_records} != {
                    PaperSource.OPENALEX,
                    PaperSource.CROSSREF,
                }:
                    raise ValueError("Same-DOI merge failed or discarded a provider record")
                Paper.model_validate_json(merged[0].model_dump_json())
                case.update(status="passed", merged_paper=merged[0].model_dump(mode="json"))
                print(f"PASS real cross-provider merge: 2 -> 1, DOI {original.doi}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep evidence even if API is unavailable
                record_failure(case, exc)
    finally:
        client.session.close()
        report.update(
            finished_at=datetime.now(UTC).isoformat(),
            all_passed=bool(report["cases"])
            and all(case["status"] == "passed" for case in report["cases"]),
        )
        target = destination / "verification.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evidence: {target}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        help="Repeat to choose sources; default tests all three",
    )
    parser.add_argument("--query", action="append", help="Repeat to test multiple queries")
    parser.add_argument("--limit", type=int, default=3, choices=range(1, 11), metavar="1..10")
    raise SystemExit(asyncio.run(verify(parser.parse_args())))
