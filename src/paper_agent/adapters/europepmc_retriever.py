"""Europe PMC retriever (Group 1, B side).

Free REST API, no key, generous limits. Biomedical focus complements the
CS-heavy sources (arXiv/Crossref/OpenAlex): EEG, medical imaging, clinical
NLP and similar health-related research ideas get first-class coverage here.

BibTeX strategy follows the other JSON-first sources: synthesize the entry
locally from the already-fetched payload (one request per paper is avoided).
"""

from __future__ import annotations

import asyncio
import re
from html import unescape
from typing import Any

from paper_agent.adapters._retriever_utils import (
    build_bibtex,
    http_get_json,
    make_bibtex_key,
    normalize_doi,
    parse_date,
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

_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePmcRetriever:
    source_name = "europe_pmc"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def search(self, plan: SearchPlan) -> list[Paper]:
        query = next((q for q in plan.queries if q.source == PaperSource.EUROPE_PMC), None)
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
        # resultType=core returns full metadata (abstract, DOI, dates) in one
        # request; the default "lite" payload would force a per-paper fetch.
        expression = sanitize_query(raw_query)
        if (filters or {}).get("date_from") or (filters or {}).get("date_to"):
            start = (filters or {}).get("date_from") or "0001-01-01"
            end = (filters or {}).get("date_to") or "9999-12-31"
            expression = f"({expression}) AND FIRST_PDATE:[{start} TO {end}]"
        if sort_by == "publication_date":
            expression += " sort_date:y"
        return http_get_json(
            _API_URL,
            params={
                "query": expression,
                "format": "json",
                "resultType": "core",
                "pageSize": max(1, min(limit, 100)),
            },
            timeout=self.timeout,
        )

    def _to_papers(self, payload: Any) -> list[Paper]:
        if not payload:
            return []
        papers: list[Paper] = []
        for item in payload.get("resultList", {}).get("result", []):
            paper = self._to_paper(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        # resultType=core includes structured author affiliations and journalInfo.
        authors: list[Author] = []
        raw_authors = (item.get("authorString") or "").strip().rstrip(".")
        for author in (item.get("authorList") or {}).get("author", []):
            name = author.get("fullName") or " ".join(
                filter(None, [author.get("firstName"), author.get("lastName")])
            )
            affiliations = [
                a["affiliation"]
                for a in (author.get("authorAffiliationDetailsList") or {}).get(
                    "authorAffiliation", []
                )
                if a.get("affiliation")
            ]
            if name:
                authors.append(Author(name=name, affiliations=affiliations))
        if not authors and raw_authors:
            for name in raw_authors.split(","):
                name = name.strip()
                if name:
                    authors.append(Author(name=name))

        publication_date = parse_date(item.get("firstPublicationDate"))
        year = item.get("pubYear")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        if publication_date:
            year = publication_date.year

        doi = normalize_doi(item.get("doi"))
        pmid = item.get("pmid")
        journal_info = item.get("journalInfo") or {}
        journal = journal_info.get("journal") or {}
        venue_name = (journal.get("title") or item.get("journalTitle") or "").strip() or None
        pmcid = item.get("pmcid")
        landing_page = (
            f"https://europepmc.org/article/PMC/{pmcid}"
            if pmcid
            else f"https://europepmc.org/article/MED/{pmid}"
            if pmid
            else None
        )

        # Abstracts may carry HTML tags (e.g. <p>, <italic>); strip to text.
        abstract = (
            " ".join(unescape(re.sub(r"<[^>]+>", " ", item.get("abstractText") or "")).split())
            or None
        )

        cited_by = item.get("citedByCount")
        try:
            cited_by = int(cited_by) if cited_by is not None else None
        except (TypeError, ValueError):
            cited_by = None

        bibtex = build_bibtex(
            key=make_bibtex_key(title, year),
            title=title,
            authors=[a.name for a in authors],
            year=year,
            venue=venue_name,
            doi=doi,
            url=landing_page,
        )

        return Paper(
            title=title,
            authors=authors,
            institutions=list(dict.fromkeys(i for a in authors for i in a.affiliations)),
            publication_date=publication_date,
            year=year,
            abstract=abstract or None,
            doi=doi,
            pmid=pmid,
            venue=Venue(
                name=venue_name,
                venue_type="journal",
                volume=journal_info.get("volume"),
                issue=journal_info.get("issue"),
            )
            if venue_name
            else None,
            citation_count=cited_by,
            is_open_access=item.get("isOpenAccess") == "Y",
            open_access_url=landing_page if item.get("isOpenAccess") == "Y" else None,
            fulltext=FullTextAsset(
                pdf_url=f"https://europepmc.org/articles/{pmcid}?pdf=render"
                if pmcid and item.get("isOpenAccess") == "Y"
                else None
            ),
            bibtex=bibtex,
            landing_page_url=landing_page,
            source_records=[
                SourceRecord(
                    source=PaperSource.EUROPE_PMC,
                    source_id=pmcid or pmid or doi or title,
                    raw_metadata={"pmcid": pmcid, "pmid": pmid, "citation_count": cited_by},
                )
            ],
        )
