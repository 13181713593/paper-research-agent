"""arXiv metadata retrieval adapter.

The public ``arxiv`` package is synchronous and exposes a lazy generator.  The
adapter keeps both the request and generator iteration off the event loop, then
maps every result through the shared BibTeX mapper.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from itertools import islice
from typing import Any

import arxiv
import bibtexparser
import requests
from bibtexparser.bibdatabase import BibDatabase

from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import Paper, PaperSource, SearchPlan, SearchQuery

from .bibtex_mapper import BibTexMapper, BibTexMappingError
from .retrieval_common import normalize_arxiv, normalize_doi


class ArxivError(ProviderUnavailableError):
    """Base class for errors returned by the arXiv adapter."""


class ArxivTimeoutError(ArxivError):
    """The arXiv endpoint timed out."""


class ArxivRateLimitError(ArxivError):
    """The arXiv endpoint rejected a request due to rate limiting."""


class ArxivNoResultsError(ArxivError):
    """A valid query returned no entries."""


@dataclass(frozen=True)
class ArxivSearchConfig:
    """Runtime policy for an arXiv search.

    ``max_retries`` is the number of retries after the initial request.  The
    arXiv service requires a minimum three second interval between requests;
    callers using the real client cannot lower that value.
    """

    page_size: int = 100
    delay_seconds: float = 3.0
    max_retries: int = 2
    timeout_connect: float = 10.0
    timeout_read: float = 30.0
    empty_results: str = "return"
    rate_limit_backoff: float = 30.0
    max_retry_wait: float = 120.0

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout_connect <= 0 or self.timeout_read <= 0:
            raise ValueError("timeouts must be positive")
        if self.empty_results not in {"error", "return"}:
            raise ValueError("empty_results must be 'error' or 'return'")
        for name in (
            "delay_seconds",
            "timeout_connect",
            "timeout_read",
            "rate_limit_backoff",
            "max_retry_wait",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.rate_limit_backoff < 3:
            raise ValueError("rate_limit_backoff must be at least three seconds")
        if self.max_retry_wait < self.rate_limit_backoff:
            raise ValueError("max_retry_wait must cover rate_limit_backoff")


class _RateGate:
    """Process-wide request gate shared by every retriever instance."""

    lock = threading.Lock()
    last_started: float | None = None
    cooldown_until: float = 0.0

    @classmethod
    def wait(cls, delay: float, max_wait: float = 120.0) -> None:
        started = time.monotonic()
        while True:
            with cls.lock:
                now = time.monotonic()
                earliest = cls.cooldown_until
                if cls.last_started is not None:
                    earliest = max(earliest, cls.last_started + delay)
                remaining = earliest - now
                if remaining <= 0:
                    cls.last_started = now
                    return
                if now - started + remaining > max_wait:
                    raise ArxivRateLimitError(
                        f"arXiv cooldown requires {remaining:.1f}s; wait budget is {max_wait:.1f}s"
                    )
            # Release the lock while waiting, then recheck for another instance's
            # newer 429. A rejected request delays every instance in this process.
            time.sleep(remaining)

    @classmethod
    def defer(cls, delay: float) -> None:
        with cls.lock:
            cls.cooldown_until = max(cls.cooldown_until, time.monotonic() + delay)


class _BoundedArxivClient(arxiv.Client):
    """Avoid the SDK's default 100-row overfetch for small searches."""

    def _format_url(self, search: arxiv.Search, start: int, page_size: int) -> str:
        if search.max_results is not None:
            page_size = min(page_size, max(1, search.max_results - start))
        return super()._format_url(search, start, page_size)


class ArxivRetriever:
    source_name = PaperSource.ARXIV.value

    def __init__(
        self,
        mapper: BibTexMapper | None = None,
        client: Any | None = None,
        config: ArxivSearchConfig | None = None,
    ) -> None:
        self.config = config or ArxivSearchConfig()
        self._external_client = client is not None
        if not self._external_client and self.config.delay_seconds < 3.0:
            raise ValueError("real arXiv clients must use delay_seconds >= 3")
        self.mapper = mapper or BibTexMapper()
        self.client = client or _BoundedArxivClient(
            page_size=self.config.page_size,
            # Rate limiting and retries are handled here so that Retry-After is
            # respected consistently.  The SDK remains the wire/parser layer.
            delay_seconds=0,
            num_retries=0,
        )
        self._last_response: Any | None = None
        if not self._external_client:
            self._install_response_capture()

    def _install_response_capture(self) -> None:
        session = getattr(self.client, "_session", None)
        if session is None or getattr(session, "_paper_agent_capture", False):
            return
        original_send = session.send

        def send(request: Any, **kwargs: Any) -> Any:
            _RateGate.wait(self.config.delay_seconds, self.config.max_retry_wait)
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (self.config.timeout_connect, self.config.timeout_read)
            self._last_response = None
            response = original_send(request, **kwargs)
            self._last_response = response
            if response.status_code in {429, 503}:
                delay = self._retry_after()
                if delay is None and response.status_code == 429:
                    delay = self.config.rate_limit_backoff
                if delay is not None:
                    _RateGate.defer(delay)
            return response

        session.send = send  # type: ignore[method-assign]
        session._paper_agent_capture = True

    async def search(self, plan: SearchPlan) -> list[Paper]:
        queries = [q for q in plan.queries if q.source == PaperSource.ARXIV]
        if not queries:
            return []
        output: list[Paper] = []
        seen: set[str] = set()
        for query in queries:
            search_query = self._build_query(query)
            rows = (
                await self._fetch(search_query, query.limit, query.sort_by)
                if query.sort_by != "relevance"
                else await self._fetch(search_query, query.limit)
            )
            if not rows:
                if self.config.empty_results == "return":
                    continue
                raise ArxivNoResultsError(f"arXiv query returned no results: {query.query!r}")
            for row in rows:
                try:
                    paper = await self._map_result(row)
                    # arXiv IDs are the authoritative identity for this source;
                    # DOI is a fallback for records where arXiv omitted it.
                    identities = (
                        [paper.arxiv_id] if paper.arxiv_id else ([paper.doi] if paper.doi else [])
                    )
                    if identities and identities[0] in seen:
                        continue
                    seen.update(identities)
                    output.append(paper)
                except BibTexMappingError as exc:
                    raise ProviderUnavailableError(
                        f"arXiv returned malformed metadata: {exc}"
                    ) from exc
        return output

    def _build_query(self, query: SearchQuery) -> str:
        raw = " ".join(str(query.query or "").split())
        if not raw:
            raise ProviderUnavailableError("arXiv query is blank")
        filters = query.filters or {}
        allowed = {"date_from", "date_to", "excluded_terms", "required_terms"}
        unknown = sorted(set(filters) - allowed)
        if unknown:
            raise ProviderUnavailableError(f"unsupported arXiv filters: {', '.join(unknown)}")
        terms = re.findall(r'"[^"]+"|\S+', raw)
        terms = [term.strip('"') for term in terms if term.strip('"')]
        if not terms:
            raise ProviderUnavailableError("arXiv query is blank")
        # Standard single-word form avoids unnecessary phrase quoting; actual
        # phrases and punctuation still need quotes to preserve their meaning.
        expression = " AND ".join(
            f"all:{term}" if re.fullmatch(r"\w+", term) else f'all:"{term.replace(chr(34), "")}"'
            for term in terms
        )
        result = f"({expression})" if len(terms) > 1 else expression

        required = filters.get("required_terms") or []
        for term in required:
            value = self._term(term)
            result += f' AND all:"{value}"'
        date_from = self._date_filter(filters.get("date_from"), "date_from")
        date_to = self._date_filter(filters.get("date_to"), "date_to")
        if date_from and date_to and date_from > date_to:
            raise ProviderUnavailableError("date_from must not be after date_to")
        if date_from or date_to:
            result += (
                f" AND submittedDate:[{date_from or '000101010000'} TO {date_to or '999912312359'}]"
            )
        for term in filters.get("excluded_terms") or []:
            result += f' ANDNOT all:"{self._term(term)}"'
        return result

    @staticmethod
    def _term(value: Any) -> str:
        value = " ".join(str(value or "").split()).replace('"', "")
        if not value:
            raise ProviderUnavailableError("arXiv filter term is blank")
        return value

    @staticmethod
    def _date_filter(value: Any, name: str) -> str | None:
        if value is None or value == "":
            return None
        try:
            if isinstance(value, datetime):
                value = value.date()
            if isinstance(value, date):
                parsed = value
            else:
                parsed = date.fromisoformat(str(value))
            return parsed.strftime("%Y%m%d0000" if name == "date_from" else "%Y%m%d2359")
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError(f"invalid {name}: {value!r}") from exc

    async def _fetch(self, query: str, limit: int, sort_by="relevance") -> list[Any]:
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate
            if sort_by == "publication_date"
            else arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await asyncio.to_thread(self._fetch_sync, search)
            except Exception as exc:  # SDK has several provider-specific exception types.
                last_error = exc
                if attempt + 1 >= attempts or not self._retryable(exc):
                    raise self._translate_error(exc) from exc
                status = getattr(exc, "status", None)
                delay = self._retry_after() if status in {429, 503} else None
                if delay is None:
                    base = self.config.rate_limit_backoff if status == 429 else 3.0
                    delay = min(base * 2**attempt, self.config.max_retry_wait)
                if delay > self.config.max_retry_wait:
                    raise self._translate_error(exc) from exc
                if status in {429, 503}:
                    _RateGate.defer(delay)
                if delay > 0:
                    await asyncio.to_thread(time.sleep, delay)
        raise ProviderUnavailableError("arXiv request failed") from last_error

    def _fetch_sync(self, search: Any) -> list[Any]:
        if self._external_client:
            _RateGate.wait(self.config.delay_seconds, self.config.max_retry_wait)
        # Iterating here is essential: Client.results is lazy and performs I/O
        # while the generator advances.
        return list(islice(self.client.results(search), search.max_results))

    def _retry_after(self) -> float | None:
        headers = getattr(self._last_response, "headers", None) or {}
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if value is None:
            return None
        try:
            seconds = float(value)
            return seconds if math.isfinite(seconds) and seconds >= 0 else None
        except (TypeError, ValueError):
            try:
                target = parsedate_to_datetime(str(value))
                if target.tzinfo is None:
                    return None
                reference = datetime.fromtimestamp(time.time(), UTC)
                # A server Date header avoids the workstation's clock skew.
                try:
                    server_date = parsedate_to_datetime(headers.get("Date", ""))
                    if server_date.tzinfo is not None:
                        reference = server_date
                except (TypeError, ValueError, OverflowError):
                    pass
                return max(0.0, (target - reference).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                TimeoutError,
                requests.Timeout,
                requests.ConnectionError,
                arxiv.UnexpectedEmptyPageError,
            ),
        ) or getattr(exc, "status", None) in {
            429,
            500,
            502,
            503,
            504,
        }

    @staticmethod
    def _translate_error(exc: Exception) -> ArxivError:
        if isinstance(exc, (ArxivError,)):
            return exc
        if isinstance(exc, (TimeoutError, requests.Timeout, requests.ConnectionError)):
            return ArxivTimeoutError(str(exc) or "arXiv request timed out")
        status = getattr(exc, "status", None)
        if status == 429:
            return ArxivRateLimitError("arXiv rate limit (HTTP 429)")
        if status is not None:
            return ProviderUnavailableError(f"arXiv request failed (HTTP {status})")
        return ArxivError(str(exc) or "arXiv request failed")

    async def _map_result(self, result: Any) -> Paper:
        entry_id = self._value(result, "entry_id") or self._value(result, "id")
        arxiv_id = self._arxiv_id(entry_id)
        title = self._value(result, "title")
        if not title or not str(title).strip() or str(title).strip() == "0":
            raise BibTexMappingError("arXiv result title is missing")
        authors = []
        for author in self._value(result, "authors", []) or []:
            name = self._value(author, "name")
            if name:
                authors.append({"name": str(name).strip()})
        published = self._value(result, "published")
        updated = self._value(result, "updated")
        doi = self._value(result, "doi")
        pdf_url = self._value(result, "pdf_url")
        source_url = entry_id or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
        source_id = arxiv_id or source_url
        if not source_id:
            raise BibTexMappingError("arXiv result identifier is missing")
        bib = self._export_bibtex(result, arxiv_id, source_url, authors)
        metadata = {
            "source_id": source_id,
            "source_url": source_url,
            "arxiv_id": arxiv_id,
            "abstract": self._value(result, "summary"),
            "authors": authors,
            "published": self._iso(published),
            "updated": self._iso(updated),
            "doi": doi,
            "pdf_url": pdf_url,
            "comments": self._value(result, "comment"),
            "venue": {"name": self._value(result, "journal_ref")}
            if self._value(result, "journal_ref")
            else None,
            "raw_metadata": self._raw_result(result),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return await self.mapper.to_paper(
            bib, source=PaperSource.ARXIV.value, supplemental_metadata=metadata
        )

    @classmethod
    def _export_bibtex(cls, result, arxiv_id, source_url, authors) -> str:
        """The SDK has no BibTeX exporter; serialize only available API fields."""

        def latex_text(value):
            # Preserve math/TeX commands, escape bare text punctuation for citations.
            parts = re.split(r"(\$[^$]*\$)", " ".join(str(value).split()))
            return "".join(
                part if i % 2 else re.sub(r"(?<!\\)([&%#_])", r"\\\1", part)
                for i, part in enumerate(parts)
            )

        entry = {
            "ENTRYTYPE": "misc",
            "ID": "arxiv_" + re.sub(r"[^A-Za-z0-9_-]", "_", arxiv_id or source_url),
            "title": "{" + latex_text(cls._value(result, "title")) + "}",
            "url": source_url,
        }
        if authors:
            entry["author"] = " and ".join(latex_text(author["name"]) for author in authors)
        identifier = normalize_arxiv(arxiv_id)
        if identifier:
            entry.update(eprint=identifier, archiveprefix="arXiv")
        primary = cls._value(result, "primary_category")
        if primary:
            entry["primaryclass"] = str(primary)
        doi = normalize_doi(cls._value(result, "doi"))
        if doi:
            entry["doi"] = doi
        published = cls._value(result, "published")
        try:
            if isinstance(published, datetime):
                published = published.date()
            elif not isinstance(published, date):
                published = datetime.fromisoformat(str(published)).date()
            if 1600 <= published.year <= 2200:
                entry.update(year=str(published.year), date=published.isoformat())
        except (TypeError, ValueError):
            # The mapper diagnoses invalid raw dates using supplemental metadata.
            pass
        database = BibDatabase()
        database.entries = [entry]
        return bibtexparser.dumps(database).strip()

    @staticmethod
    def _value(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _arxiv_id(value: Any) -> str | None:
        if not value:
            return None
        text = str(value).rstrip("/")
        match = re.search(r"(?:abs|pdf)/(.*)$", text, re.IGNORECASE)
        return (match.group(1) if match else text).removesuffix(".pdf")

    @classmethod
    def _raw_result(cls, result: Any) -> dict[str, Any]:
        def safe(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            if isinstance(value, dict):
                return {str(key): safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [safe(item) for item in value]
            if hasattr(value, "__dict__"):
                return {str(key): safe(item) for key, item in vars(value).items()}
            return str(value)

        values = vars(result) if hasattr(result, "__dict__") else {}
        return {str(key): safe(value) for key, value in values.items()}
