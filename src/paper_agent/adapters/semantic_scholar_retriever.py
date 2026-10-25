"""Semantic Scholar retriever (Group 1, B side).

Unauthenticated tier allows ~100 requests / 5 minutes. 429 responses are
retried with exponential backoff in the shared HTTP layer; a persistent 429
surfaces as ``ProviderRateLimitedError`` which the pipeline converts into a
per-source warning instead of failing the run.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from paper_agent.adapters._retriever_utils import (
    build_bibtex,
    http_get_json,
    make_bibtex_key,
    normalize_doi,
    parse_date,
    sanitize_query,
)
from paper_agent.domain.models import Author, Paper, PaperSource, SearchPlan, SourceRecord, Venue

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = (
    "paperId,title,abstract,year,publicationDate,authors,venue,"
    "externalIds,citationCount,openAccessPdf"
)


class SemanticScholarRetriever:
    source_name = "semantic_scholar"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        # Optional key lifts the shared unauthenticated pool (~100 req/5 min)
        # to a dedicated limit. Never hardcode; read from the environment.
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
        self.timeout = timeout

    async def search(self, plan: SearchPlan) -> list[Paper]:
        query = next((q for q in plan.queries if q.source == PaperSource.SEMANTIC_SCHOLAR), None)
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
        headers = {"x-api-key": self.api_key} if self.api_key else None
        # Public pool is strict (shared ~100 req/5min): start with a longer
        # backoff and keep retrying while honoring the Retry-After header.
        params = {
            "query": sanitize_query(raw_query),
            "limit": max(1, min(limit, 100)),
            "fields": _FIELDS,
        }
        if (filters or {}).get("date_from") or (filters or {}).get("date_to"):
            params["publicationDateOrYear"] = (
                f"{(filters or {}).get('date_from') or ''}:{(filters or {}).get('date_to') or ''}"
            )
        url = _API_URL
        if sort_by == "publication_date":
            url += "/bulk"
            params.pop("limit", None)  # Bulk uses fixed server batches; take only our quota below.
            params["sort"] = "publicationDate:desc"
            terms = sanitize_query(raw_query).replace('"', "")
            params["query"] = f'"{terms}"'
        payload = http_get_json(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
            max_retries=1,
            initial_backoff=1.0,
            max_retry_after=5.0,
        )
        if sort_by == "publication_date" and isinstance(payload, dict):
            payload = {**payload, "data": payload.get("data", [])[:limit]}
        return payload

    def _to_papers(self, payload: Any) -> list[Paper]:
        if not payload:
            return []
        papers: list[Paper] = []
        for item in payload.get("data", []):
            paper = self._to_paper(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        authors = [Author(name=a.get("name")) for a in item.get("authors", []) if a.get("name")]
        publication_date = parse_date(item.get("publicationDate"))
        year = item.get("year")
        if publication_date:
            year = publication_date.year

        external = item.get("externalIds") or {}
        doi = normalize_doi(external.get("DOI"))
        arxiv_id = external.get("ArXiv")
        paper_id = item.get("paperId")
        venue_name = (item.get("venue") or "").strip() or None
        # Some records carry openAccessPdf with an empty url: contract says
        # unknown/empty scalars become None, never "" (URL validation would fail).
        open_access_pdf = (item.get("openAccessPdf") or {}).get("url") or None
        url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None

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
            publication_date=publication_date,
            year=year,
            abstract=(item.get("abstract") or None),
            arxiv_id=arxiv_id,
            doi=doi,
            pmid=external.get("PubMed"),
            semantic_scholar_id=paper_id,
            venue=Venue(name=venue_name) if venue_name else None,
            citation_count=item.get("citationCount"),
            is_open_access=bool(open_access_pdf),
            open_access_url=open_access_pdf,
            bibtex=bibtex,
            landing_page_url=url,
            source_records=[
                SourceRecord(
                    source=PaperSource.SEMANTIC_SCHOLAR,
                    source_id=paper_id or doi or title,
                    raw_metadata={},
                )
            ],
        )
