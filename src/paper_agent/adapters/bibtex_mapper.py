"""Convert exactly one BibTeX entry using explicit, auditable provider metadata."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import splitname
from bibtexparser.latexenc import latex_to_unicode
from pydantic import HttpUrl, TypeAdapter

from paper_agent.domain.errors import PaperAgentError
from paper_agent.domain.models import (
    Author,
    FullTextAsset,
    Paper,
    PaperSource,
    PublicationStatus,
    SourceRecord,
    Venue,
)

from .retrieval_common import normalize_arxiv, normalize_doi, normalize_title, text


class BibTexMappingError(PaperAgentError):
    """Input cannot yield one valid Paper; callers must not silently drop it."""


def _decode(value):
    if value is None:
        return None
    maths = []

    def keep(match):
        maths.append(match.group(0))
        return f"MATHPLACEHOLDER{len(maths) - 1}END"

    value = re.sub(r"\$[^$]*\$", keep, str(value))
    value = latex_to_unicode(value)
    for i, part in enumerate(maths):
        value = value.replace(f"MATHPLACEHOLDER{i}END", part)
    return text(value)


def _author_tokens(value):
    # BibTeX's separator is a top-level 'and', not 'and' inside a corporate name.
    depth = 0
    start = 0
    for m in re.finditer(r"(?<!\\)[{}]|\s+and\s+", value, re.IGNORECASE):
        token = m.group()
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        elif depth == 0:
            yield value[start : m.start()].strip()
            start = m.end()
    yield value[start:].strip()


def _authors(value):
    if not value:
        return []
    result = []
    for token in _author_tokens(value):
        if not token:
            continue
        if token.startswith("{") and token.endswith("}"):
            name = _decode(token)
        else:
            parts = splitname(token, strict_mode=False)
            name = _decode(
                " ".join(parts.get("first", []) + parts.get("von", []) + parts.get("last", []))
            )
            if parts.get("jr"):
                name += ", " + (_decode(" ".join(parts["jr"])) or "")
        if name:
            result.append(Author(name=name, normalized_name=normalize_title(name)))
    return result


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Not JSON-serializable: {type(value).__name__}")


class BibTexMapper:
    async def to_paper(
        self,
        bibtex: str,
        *,
        source: str,
        supplemental_metadata: dict[str, Any] | None = None,
    ) -> Paper:
        try:
            return self._convert(bibtex, source, supplemental_metadata or {})
        except BibTexMappingError:
            raise
        except Exception as exc:
            raise BibTexMappingError(f"{source}: invalid BibTeX/metadata: {exc}") from exc

    def _convert(self, bibtex, source, supplemental):
        provider = PaperSource(source)
        # Copy/validate provenance up front; do not retain SDK instances or mutate callers.
        extra = json.loads(json.dumps(supplemental, default=_json_default, allow_nan=False))
        parser = BibTexParser(common_strings=True)
        # Crossref exports full unquoted month names (month=June), while the
        # parser only predefines abbreviations. Keep raw BibTeX untouched;
        # explicit @string definitions in the input can still override these.
        parser.bib_database.strings.update(
            {month.casefold(): month for month in list(parser.bib_database.strings.values())}
        )
        parser.ignore_nonstandard_types = False
        parser.homogenize_fields = False
        database = bibtexparser.loads(bibtex, parser=parser)
        if len(database.entries) != 1:
            raise BibTexMappingError("Expected exactly one valid BibTeX entry")
        entry = database.entries[0]
        conflicts = []

        def choose(key, fallback=None):
            return extra[key] if extra.get(key) is not None else fallback

        def diagnose(field, value):
            conflicts.append(f"Invalid {field}: {value!r}; canonical value left unknown")

        def url(field, value):
            if not text(value):
                return None
            try:
                return TypeAdapter(HttpUrl).validate_python(value)
            except ValueError:
                diagnose(field, value)
                return None

        title = text(choose("title")) or _decode(entry.get("title"))
        if not title:
            raise BibTexMappingError("Paper title is missing or blank")
        year_value = choose("year", entry.get("year"))
        year = None
        if year_value is not None:
            if re.fullmatch(r"\d{4}", str(year_value)) and 1600 <= int(year_value) <= 2200:
                year = int(year_value)
            else:
                diagnose("year", year_value)
        date_value = choose("publication_date", choose("published", entry.get("date")))
        published = None
        if date_value:
            try:
                if isinstance(date_value, datetime):
                    published = date_value.date()
                elif isinstance(date_value, date):
                    published = date_value
                else:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", str(date_value)):
                        raise ValueError("Not a complete ISO date")
                    published = datetime.fromisoformat(str(date_value)).date()
                if not 1600 <= published.year <= 2200:
                    raise ValueError("Date out of supported range")
            except ValueError:
                published = None
                diagnose("publication_date", date_value)
        if published:
            if year is not None and year != published.year:
                conflicts.append(
                    f"year: BibTeX/metadata {year}; full date {published}; chose date year"
                )
            year = published.year

        authors = _authors(entry.get("author"))
        if extra.get("authors"):
            authors = []
            for author in extra["authors"]:
                if isinstance(author, str):
                    authors.append(Author(name=author))
                elif isinstance(author, dict):
                    author_fields = author.copy()
                    if author_fields.get("affiliations") is None:
                        author_fields["affiliations"] = []
                    authors.append(Author.model_validate(author_fields))
                else:
                    raise BibTexMappingError("authors must contain strings or Author dictionaries")
        doi_value = choose("doi", entry.get("doi"))
        doi = normalize_doi(doi_value)
        if text(doi_value) and not doi:
            diagnose("doi", doi_value)
        eprint = (
            entry.get("eprint")
            if provider == PaperSource.ARXIV or entry.get("archiveprefix", "").lower() == "arxiv"
            else None
        )
        arxiv_value = choose("arxiv_id", eprint)
        arxiv_id = normalize_arxiv(arxiv_value)
        if text(arxiv_value) and not arxiv_id:
            diagnose("arxiv_id", arxiv_value)
        version = re.search(r"v\d+$", arxiv_id or "")
        venue = None
        if entry.get("journal") or entry.get("booktitle"):
            venue = Venue(
                name=_decode(entry.get("journal") or entry.get("booktitle")),
                venue_type="journal" if entry.get("journal") else "conference",
                publisher=_decode(entry.get("publisher")),
                volume=text(entry.get("volume")),
                issue=text(entry.get("number")),
                pages=text(entry.get("pages")),
            )
        if extra.get("venue") is not None:
            supplemental_venue = Venue.model_validate(extra["venue"])
            if venue is None:
                venue = supplemental_venue
            else:
                for field in Venue.model_fields:
                    incoming = getattr(supplemental_venue, field)
                    current = getattr(venue, field)
                    if incoming is None:
                        continue
                    if current is not None and incoming != current:
                        conflicts.append(
                            f"venue.{field}: conflicting BibTeX and supplemental values"
                        )
                        # Keep an explicit journal/booktitle distinction even when
                        # an incomplete or conflicting provider record is supplied.
                        if field == "venue_type":
                            continue
                    setattr(venue, field, incoming)
        abstract = text(choose("abstract")) or _decode(entry.get("abstract"))
        if (
            extra.get("abstract")
            and entry.get("abstract")
            and _decode(entry["abstract"]) != abstract
        ):
            conflicts.append("abstract: chose provider supplemental text; original BibTeX retained")
        source_id = text(extra.get("source_id")) or arxiv_id or doi
        if not source_id:
            source_id = "bibtex-sha256:" + hashlib.sha256(bibtex.encode()).hexdigest()
        retrieved = extra.get("retrieved_at")
        if retrieved:
            if not isinstance(retrieved, datetime):
                retrieved = datetime.fromisoformat(str(retrieved))
            if retrieved.tzinfo is None:
                raise BibTexMappingError("retrieved_at must include a timezone")
        else:
            retrieved = datetime.now(UTC)
        raw_value = extra.get("raw_metadata")
        if raw_value is not None and not isinstance(raw_value, dict):
            raise BibTexMappingError("raw_metadata must be a dictionary or None")
        raw = raw_value.copy() if raw_value is not None else {}
        if any(k in raw for k in ("_bibtex", "_supplemental", "_provider_payload")):
            raw = {"_provider_payload": raw}
        raw["_bibtex"] = bibtex
        raw["_supplemental"] = {k: v for k, v in extra.items() if k != "raw_metadata"}
        landing = url(
            "landing_page_url", choose("landing_page_url", choose("source_url", entry.get("url")))
        )
        pdf = url("pdf_url", extra.get("pdf_url"))
        keywords = extra.get("keywords") or [
            v.strip() for v in entry.get("keywords", "").split(",") if v.strip()
        ]
        explicit_institutions = extra.get("institutions")
        if explicit_institutions is None:
            explicit_institutions = []
        if not isinstance(explicit_institutions, list):
            raise BibTexMappingError("institutions must be a list or None")
        institutions = list(
            dict.fromkeys(explicit_institutions + [v for a in authors for v in a.affiliations])
        )
        scalar = {}
        for name in ("openalex_id", "semantic_scholar_id", "pmid", "language"):
            scalar[name] = text(extra.get(name))
        for name in ("is_open_access", "is_retracted"):
            value = extra.get(name)
            scalar[name] = value if isinstance(value, bool) else None
            if value is not None and not isinstance(value, bool):
                diagnose(name, value)
        citations = extra.get("citation_count")
        if citations is not None and (
            isinstance(citations, bool) or not isinstance(citations, int) or citations < 0
        ):
            diagnose("citation_count", citations)
            citations = None
        status = extra.get("publication_status")
        if status is None:
            status = (
                PublicationStatus.PREPRINT
                if provider == PaperSource.ARXIV
                else PublicationStatus.UNKNOWN
            )
        return Paper(
            paper_id=uuid5(NAMESPACE_URL, f"paper-agent:{provider}:{source_id}"),
            title=title,
            normalized_title=normalize_title(title),
            authors=authors,
            institutions=institutions,
            year=year,
            publication_date=published,
            publication_status=status,
            venue=venue,
            abstract=abstract,
            keywords=keywords,
            comments=text(choose("comments", entry.get("note"))),
            doi=doi,
            arxiv_id=arxiv_id,
            bibtex=bibtex,
            citation_count=citations,
            landing_page_url=landing,
            open_access_url=url("open_access_url", extra.get("open_access_url")),
            fulltext=FullTextAsset(
                pdf_url=pdf, landing_page_url=landing, version=version.group() if version else None
            ),
            source_records=[
                SourceRecord(
                    source=provider,
                    source_id=source_id,
                    source_url=url("source_url", extra.get("source_url")) or landing,
                    retrieved_at=retrieved,
                    raw_metadata=raw,
                )
            ],
            metadata_conflicts=conflicts,
            **scalar,
        )
