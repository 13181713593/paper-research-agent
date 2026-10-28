"""OpenReview note search adapter using the public API v2 search endpoint."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
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

_API_URL = "https://api2.openreview.net/notes/search"


def _filter_date(value: Any) -> date | None:
    return parse_date(str(value)) if value else None


def _value(content: dict[str, Any], key: str, default=None):
    value = content.get(key, default)
    return value.get("value", default) if isinstance(value, dict) else value


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    result = " ".join(unescape(str(value)).split())
    return result or None


def _note_date(note: dict[str, Any]) -> date | None:
    raw = note.get("pdate") or note.get("tcdate") or note.get("cdate")
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(float(raw) / 1000, UTC).date()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class OpenReviewRetriever:
    source_name = PaperSource.OPENREVIEW.value

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def search(self, plan: SearchPlan) -> list[Paper]:
        queries = [q for q in plan.queries if q.source == PaperSource.OPENREVIEW]
        output: list[Paper] = []
        seen: set[str] = set()
        for query in queries:
            date_from = _filter_date((query.filters or {}).get("date_from"))
            date_to = _filter_date((query.filters or {}).get("date_to"))
            # OpenReview has no server-side date filter or sort, so the window is
            # applied only after fetching. Over-fetch so in-window records survive
            # the client-side filter, then cap to the caller's quota. Without this,
            # `limit` truncates the raw batch first and most in-window records are
            # discarded.
            fetch_limit = max(1, min(query.limit * 4, 100))
            payload = await asyncio.to_thread(self._fetch, query.query, fetch_limit, query.filters)
            kept = 0
            for paper in self._to_papers(payload):
                if paper.publication_date and (
                    (date_from and paper.publication_date < date_from)
                    or (date_to and paper.publication_date > date_to)
                ):
                    continue
                identity = paper.doi or str(paper.landing_page_url) or paper.title.casefold()
                if identity in seen:
                    continue
                seen.add(identity)
                output.append(paper)
                kept += 1
                if kept >= query.limit:
                    break
        return output

    def _fetch(self, raw_query: str, limit: int, filters=None) -> Any:
        params: dict[str, Any] = {
            "term": sanitize_query(raw_query).replace('"', ""),
            "content": "all",
            "group": "all",
            # source=all 会返回评审意见等回复帖（没有题名），它们会被丢弃；
            # 只取 forum 主帖才能拿到投稿的题名、摘要与 venue。
            "source": "forum",
            "limit": max(1, min(limit, 100)),
        }
        return http_get_json(_API_URL, params=params, timeout=self.timeout)

    def _to_papers(self, payload: Any) -> list[Paper]:
        notes = payload.get("notes", []) if isinstance(payload, dict) else []
        return [paper for note in notes if (paper := self._to_paper(note)) is not None]

    def _to_paper(self, note: dict[str, Any]) -> Paper | None:
        content = note.get("content") or {}
        title = _clean(_value(content, "title"))
        if not title:
            return None
        raw_authors = _value(content, "authors", []) or []
        names = [a if isinstance(a, str) else a.get("name") for a in raw_authors]
        names = [name.strip() for name in names if isinstance(name, str) and name.strip()]
        raw_affiliations = (
            _value(content, "affiliations", []) or _value(content, "institution", []) or []
        )
        if isinstance(raw_affiliations, str):
            raw_affiliations = [raw_affiliations]
        affiliations = [a.strip() for a in raw_affiliations if isinstance(a, str) and a.strip()]
        authors = [Author(name=name, affiliations=affiliations) for name in names]
        publication_date = _note_date(note)
        venue_name = _clean(_value(content, "venue"))
        doi = normalize_doi(_clean(_value(content, "doi")))
        note_id = str(note.get("id") or note.get("forum") or title)
        landing = f"https://openreview.net/forum?id={note_id}"
        abstract = _clean(_value(content, "abstract"))
        keywords = _value(content, "keywords", []) or []
        if isinstance(keywords, str):
            keywords = [keywords]
        year = publication_date.year if publication_date else None
        bibtex = build_bibtex(
            key=make_bibtex_key(title, year),
            title=title,
            authors=names,
            year=year,
            venue=venue_name,
            doi=doi,
            url=landing,
        )
        return Paper(
            title=title,
            authors=authors,
            institutions=affiliations,
            publication_date=publication_date,
            year=year,
            venue=Venue(name=venue_name, venue_type="conference") if venue_name else None,
            abstract=abstract,
            keywords=[str(k) for k in keywords if k],
            doi=doi,
            landing_page_url=landing,
            bibtex=bibtex,
            fulltext=FullTextAsset(landing_page_url=landing),
            source_records=[
                SourceRecord(
                    source=PaperSource.OPENREVIEW,
                    source_id=note_id,
                    source_url=landing,
                    raw_metadata={"note_id": note_id},
                )
            ],
        )
