"""OpenAlex retriever (Group 1, B side).

基础调用可匿名；配置 OPENALEX_API_KEY 时使用免费账号额度。
PAPER_AGENT_CONTACT_EMAIL 作为可选联系邮箱发送。

The only data-transformation difficulty among our sources: OpenAlex returns
abstracts as an inverted word-position index (``abstract_inverted_index``),
rebuilt here into normal text. Missing abstracts stay ``None``.
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
from paper_agent.domain.models import (
    Author,
    FullTextAsset,
    Paper,
    PaperSource,
    SearchPlan,
    SourceRecord,
    Venue,
)

_API_URL = "https://api.openalex.org/works"


def _openalex_id_from_url(id_url: str | None) -> str | None:
    """https://openalex.org/W12345 -> W12345"""
    if not id_url:
        return None
    return id_url.rsplit("/", 1)[-1] or None


class OpenAlexRetriever:
    source_name = "openalex"

    def __init__(
        self,
        contact_email: str | None = None,
        timeout: float = 30.0,
        *,
        api_key: str | None = None,
    ):
        self.contact_email = contact_email or os.getenv("PAPER_AGENT_CONTACT_EMAIL", "")
        self.timeout = timeout
        self._api_key = (os.getenv("OPENALEX_API_KEY", "") if api_key is None else api_key).strip()

    async def search(self, plan: SearchPlan) -> list[Paper]:
        query = next((q for q in plan.queries if q.source == PaperSource.OPENALEX), None)
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
        params = {
            "search": sanitize_query(raw_query),
            "per-page": max(1, min(limit, 200)),
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        dates = []
        if (filters or {}).get("date_from"):
            dates.append("from_publication_date:" + str(filters["date_from"]))
        if (filters or {}).get("date_to"):
            dates.append("to_publication_date:" + str(filters["date_to"]))
        if dates:
            params["filter"] = ",".join(dates)
        if sort_by == "publication_date":
            params["sort"] = "publication_date:desc"
        if self._api_key:
            return http_get_json(
                _API_URL,
                params=params,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return http_get_json(_API_URL, params=params, timeout=self.timeout)

    def _to_papers(self, payload: Any) -> list[Paper]:
        if not payload:
            return []
        papers: list[Paper] = []
        for item in payload.get("results", []):
            paper = self._to_paper(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        title = (item.get("display_name") or item.get("title") or "").strip()
        if not title:
            return None

        authors: list[Author] = []
        institutions: list[str] = []
        for authorship in item.get("authorships", []):
            author_info = authorship.get("author") or {}
            name = (author_info.get("display_name") or "").strip()
            if not name:
                continue
            affiliations = [
                i.get("display_name")
                for i in authorship.get("institutions", [])
                if i.get("display_name")
            ]
            authors.append(
                Author(name=name, orcid=_orcid(author_info.get("orcid")), affiliations=affiliations)
            )
            institutions.extend(affiliations)

        publication_date = parse_date(item.get("publication_date"))
        year = item.get("publication_year")
        if publication_date:
            year = publication_date.year

        doi = normalize_doi(item.get("doi"))
        ids = item.get("ids") or {}
        openalex_id = _openalex_id_from_url(ids.get("openalex")) or _openalex_id_from_url(
            item.get("id")
        )
        oa = item.get("open_access") or {}

        primary_location = item.get("primary_location") or {}
        source_info = primary_location.get("source") or {}
        venue_name = source_info.get("display_name")
        landing_page = primary_location.get("landing_page_url") or item.get("id")
        locations = [
            item.get("best_oa_location") or {},
            primary_location,
            *(item.get("locations") or []),
        ]
        pdfs = list(
            dict.fromkeys(
                loc.get("pdf_url")
                for loc in locations
                if isinstance(loc, dict) and loc.get("pdf_url")
            )
        )

        abstract = _restore_abstract(item.get("abstract_inverted_index"))
        if abstract is None:
            abstract = item.get("abstract") or None

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
            institutions=institutions,
            publication_date=publication_date,
            year=year,
            abstract=abstract,
            doi=doi,
            openalex_id=openalex_id,
            venue=Venue(
                name=venue_name,
                venue_type=source_info.get("type"),
                publisher=source_info.get("host_organization_name"),
            )
            if venue_name
            else None,
            citation_count=item.get("cited_by_count"),
            is_open_access=oa.get("is_oa"),
            open_access_url=(oa.get("oa_url") or None),
            fulltext=FullTextAsset(pdf_url=pdfs[0] if pdfs else None, candidate_urls=pdfs),
            bibtex=bibtex,
            landing_page_url=landing_page,
            source_records=[
                SourceRecord(
                    source=PaperSource.OPENALEX,
                    source_id=openalex_id or doi or title,
                    raw_metadata={
                        "openalex_type": (item.get("type") or ""),
                        "citation_count": item.get("cited_by_count"),
                    },
                )
            ],
        )


def _restore_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted or not isinstance(inverted, dict):
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted.items():
        for index in indices or []:
            positions[int(index)] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _orcid(orcid_url: str | None) -> str | None:
    if not orcid_url:
        return None
    return orcid_url.rsplit("/", 1)[-1] or None
