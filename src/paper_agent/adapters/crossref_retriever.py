"""Crossref retriever (Group 1, B side).

Free REST API, no key. Polite-pool usage: include a contact email in the
User-Agent when PAPER_AGENT_CONTACT_EMAIL is configured.

BibTeX strategy: the task only requires BibTeX-grade metadata, and Crossref
exposes true BibTeX only via one content-negotiation request per DOI. To keep
the pipeline fast we synthesize the BibTeX entry locally from the JSON payload
(already fetched); the string is attached to ``Paper.bibtex`` so group A's
BibTexMapper semantics stay consistent across sources.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import date
from html import unescape
from typing import Any

from paper_agent.adapters._retriever_utils import (
    build_bibtex,
    http_get_json,
    make_bibtex_key,
    normalize_doi,
    parse_year,
    sanitize_query,
)
from paper_agent.domain.models import (
    Author,
    FullTextAsset,
    Paper,
    PaperSource,
    SearchPlan,
    SourceRecord,
    Venue,
)

_API_URL = "https://api.crossref.org/works"
_SELECT_FIELDS = "DOI,title,author,container-title,issued,published,abstract,is-referenced-by-count,URL,type,link,publisher"
_JATS_TAG_RE = re.compile(r"<[^>]+>")


def _strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts are JATS XML; strip tags for the Paper model."""
    if not abstract:
        return None
    text = _JATS_TAG_RE.sub(" ", abstract)
    cleaned = " ".join(unescape(text).split())
    return cleaned or None


def _date_from_parts(parts: list[Any] | None) -> date | None:
    if not parts or not isinstance(parts, list):
        return None
    first = parts[0]
    if not isinstance(first, list) or not first:
        return None
    try:
        year = int(first[0])
        month = int(first[1]) if len(first) > 1 else 1
        day = int(first[2]) if len(first) > 2 else 1
        return date(year, month, day)
    except (ValueError, TypeError, IndexError):
        return None


class CrossrefRetriever:
    source_name = "crossref"

    def __init__(self, contact_email: str | None = None, timeout: float = 30.0):
        self.contact_email = contact_email or os.getenv("PAPER_AGENT_CONTACT_EMAIL", "")
        self.timeout = timeout

    async def search(self, plan: SearchPlan) -> list[Paper]:
        query = next((q for q in plan.queries if q.source == PaperSource.CROSSREF), None)
        if query is None:
            return []
        payload = (
            await asyncio.to_thread(
                self._fetch, query.query, query.limit, query.filters, query.sort_by
            )
            if query.filters or query.sort_by != "relevance"
            else await asyncio.to_thread(self._fetch, query.query, query.limit)
        )
        return self._to_papers(payload)

    def _fetch(self, raw_query: str, limit: int, filters=None, sort_by="relevance") -> Any:
        headers = {}
        if self.contact_email:
            headers["User-Agent"] = f"paper-research-agent/0.2 (mailto:{self.contact_email})"
        params = {
            "query.bibliographic": sanitize_query(raw_query),
            "rows": max(1, min(limit, 100)),
            "select": _SELECT_FIELDS,
        }
        dates = []
        if (filters or {}).get("date_from"):
            dates.append("from-pub-date:" + str(filters["date_from"]))
        if (filters or {}).get("date_to"):
            dates.append("until-pub-date:" + str(filters["date_to"]))
        if dates:
            params["filter"] = ",".join(dates)
        if sort_by == "publication_date":
            params.update(sort="published", order="desc")
            # Title matching reduces incidental mentions when taking newest records.
            params["query.title"] = params.pop("query.bibliographic")
        return http_get_json(
            _API_URL,
            params=params,
            headers=headers or None,
            timeout=self.timeout,
        )

    def _to_papers(self, payload: Any) -> list[Paper]:
        if not payload:
            return []
        items = payload.get("message", {}).get("items", [])
        papers: list[Paper] = []
        for item in items:
            paper = self._to_paper(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        titles = item.get("title") or []
        title = titles[0].strip() if titles and titles[0] else None
        if not title:
            return None

        authors = [
            Author(
                name=" ".join(filter(None, [a.get("given"), a.get("family")])).strip(),
                orcid=normalize_orcid(a.get("ORCID")),
                affiliations=[
                    aff.get("name") for aff in a.get("affiliation", []) if aff.get("name")
                ],
            )
            for a in item.get("author", [])
            if " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
        ]

        issued = _date_from_parts((item.get("issued") or {}).get("date-parts"))
        published = _date_from_parts((item.get("published") or {}).get("date-parts"))
        publication_date = issued or published
        year = (
            parse_year((item.get("issued") or {}).get("date-parts", [[None]])[0][0])
            if (item.get("issued") or {}).get("date-parts")
            else None
        )
        if publication_date:
            year = publication_date.year

        doi = normalize_doi(item.get("DOI"))
        venue_names = item.get("container-title") or []
        venue_name = venue_names[0] if venue_names else None
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)

        bibtex = build_bibtex(
            key=make_bibtex_key(title, year),
            title=title,
            authors=[a.name for a in authors],
            year=year,
            venue=venue_name,
            doi=doi,
            url=url,
        )

        return Paper(
            title=title,
            authors=authors,
            institutions=[i for a in authors for i in a.affiliations],
            publication_date=publication_date,
            year=year,
            abstract=_strip_jats(item.get("abstract")),
            doi=doi,
            venue=Venue(
                name=venue_name,
                publisher=item.get("publisher"),
                venue_type={"journal-article": "journal", "proceedings-article": "conference"}.get(
                    item.get("type")
                ),
            )
            if venue_name
            else None,
            citation_count=item.get("is-referenced-by-count"),
            bibtex=bibtex,
            landing_page_url=url,
            fulltext=FullTextAsset(
                candidate_urls=[
                    link["URL"]
                    for link in item.get("link", [])
                    if isinstance(link, dict)
                    and link.get("URL")
                    and (
                        "pdf" in (link.get("content-type") or "").lower()
                        or link["URL"].lower().endswith(".pdf")
                    )
                ]
            ),
            source_records=[
                SourceRecord(
                    source=PaperSource.CROSSREF,
                    source_id=doi or url or title,
                    raw_metadata={
                        "crossref_type": item.get("type"),
                        "citation_count": item.get("is-referenced-by-count"),
                    },
                )
            ],
        )


def normalize_orcid(orcid: str | None) -> str | None:
    if not orcid:
        return None
    return orcid.rsplit("/", 1)[-1] if "orcid.org" in orcid else orcid
