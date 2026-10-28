"""Shared helpers for Group 1 retrievers (B side).

Conventions from docs/API_CONTRACTS.md:
- Sync HTTP calls are wrapped with ``asyncio.to_thread`` (thread pool), as
  recommended by the architecture for sync SDKs.
- Provider failures raise typed errors; the pipeline converts them into
  per-source warnings.
- Unknown metadata becomes ``None`` — never "unknown", empty strings or
  invented values.
- Illegal dates become ``None`` instead of crashing the whole source.
"""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from paper_agent.domain.request_budget import remaining_time, wait_with_budget


class ProviderError(Exception):
    """Base class for typed provider failures raised by retrievers."""


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitedError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    """The provider answered but the payload could not be trusted."""


_HOST_LOCK = threading.Lock()
_HOST_COOLDOWNS: dict[str, float] = {}
_HOST_NEXT: dict[str, float] = {}
_HOST_FAILURES: dict[str, ProviderError] = {}


def _retry_after(response: Any, fallback: float) -> float:
    """Respect the provider's Retry-After header (seconds or HTTP-date).

    Falls back to the caller's exponential backoff when absent/unparsable and
    never shortens the server's delay. The caller stops if its budget is shorter.
    """
    headers = getattr(response, "headers", None) or {}
    getter = getattr(headers, "get", None)
    if getter is None:
        return fallback
    value = getter("Retry-After")
    if not isinstance(value, (str, int, float)) or not value:
        return fallback
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return fallback
        if when.tzinfo is None:
            return fallback
        reference = datetime.now(UTC)
        try:
            server_date = parsedate_to_datetime(str(getter("Date") or ""))
            if server_date.tzinfo is not None:
                reference = server_date
        except (ValueError, TypeError, OverflowError):
            pass
        seconds = (when - reference).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) and seconds >= 0 else fallback


def _host_gate(host, deadline):
    """Reserve a request slot and share cooldowns across simultaneous app sessions."""
    # Public services only; injected unit-test endpoints have no pacing policy.
    interval = (
        1.0
        if host
        in {"api.crossref.org", "api.openalex.org", "api.semanticscholar.org", "www.ebi.ac.uk"}
        else 0.0
    )
    while True:
        with _HOST_LOCK:
            now = time.monotonic()
            cooldown = _HOST_COOLDOWNS.get(host, 0)
            if cooldown >= deadline:
                raise _HOST_FAILURES.get(
                    host, ProviderRateLimitedError("Provider cooldown exceeds request budget")
                )
            wait = max(cooldown, _HOST_NEXT.get(host, 0)) - now
            if wait <= 0:
                _HOST_NEXT[host] = now + interval
                return
        if wait >= deadline - time.monotonic():
            raise ProviderTimeoutError("Request pacing exceeds time budget")
        wait_with_budget(wait)


def http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    max_retries: int = 2,
    initial_backoff: float = 0.5,
    max_retry_after: float = 15.0,
    total_timeout: float = 25.0,
) -> Any:
    """Blocking GET returning parsed JSON with retry/backoff.

    Retries 429 and 5xx with exponential backoff (default 0.5s, 1s, 2s). A 429
    honors the provider's ``Retry-After`` header when present. A 429 that
    persists after all retries raises ``ProviderRateLimitedError``; connection
    problems raise ``ProviderUnavailableError``.
    """
    deadline = time.monotonic() + remaining_time(total_timeout)
    host = urlsplit(url).netloc
    for attempt in range(max_retries + 1):
        _host_gate(host, deadline)
        remaining = remaining_time(deadline - time.monotonic())
        if remaining <= 0:
            raise ProviderTimeoutError("Request time budget exhausted")
        response = None
        failure = None
        backoff = random.uniform(0, min(8.0, initial_backoff * 2**attempt))
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=min(timeout, remaining)
            )
        except requests.Timeout as exc:
            failure = ProviderTimeoutError("Provider request timed out")
            failure.__cause__ = exc
        except requests.RequestException as exc:
            failure = ProviderUnavailableError("Provider connection unavailable")
            failure.__cause__ = exc
        if response is not None:
            try:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderResponseError("Provider returned invalid JSON") from exc
                if response.status_code == 429:
                    failure = ProviderRateLimitedError("Provider rate limited")
                    backoff = _retry_after(response, fallback=max(2.0, backoff))
                elif 500 <= response.status_code < 600:
                    failure = ProviderUnavailableError(
                        f"Provider server error {response.status_code}"
                    )
                    backoff = _retry_after(response, fallback=backoff)
                else:
                    raise ProviderResponseError(f"Provider returned HTTP {response.status_code}")
                with _HOST_LOCK:
                    _HOST_COOLDOWNS[host] = max(
                        _HOST_COOLDOWNS.get(host, 0), time.monotonic() + backoff
                    )
                    _HOST_FAILURES[host] = failure
            finally:
                response.close()
        if (
            attempt >= max_retries
            or backoff > max_retry_after
            or backoff >= remaining_time(deadline - time.monotonic())
        ):
            raise failure
        wait_with_budget(backoff)


def parse_date(value: str | None) -> date | None:
    """Parse ISO-ish date strings; return None for missing/illegal values."""
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_year(value: Any) -> int | None:
    """Coerce a provider year field to int; None for missing/garbage."""
    if value is None:
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1600 <= year <= 2200 else None


def restore_abstract_from_inverted_index(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild an abstract from OpenAlex's inverted word-position index.

    ``{"word": [0, 5], "next": [1]}`` -> "word next ... word".
    Returns None when the index is missing or empty, so the Paper model keeps
    its explicit unknown state instead of an empty string.
    """
    if not inverted or not isinstance(inverted, dict):
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted.items():
        for index in indices or []:
            positions[int(index)] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def normalize_doi(doi: str | None) -> str | None:
    """Strip URL prefixes so ``doi.org/10.x/y`` and ``10.x/y`` compare equal."""
    if not doi:
        return None
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi or None


def make_bibtex_key(title: str, year: int | None) -> str:
    """Deterministic citation key from the first title word + year."""
    first = next((w for w in title.split() if w.isalpha()), "paper")
    return f"{first.lower()}{year or ''}"


def build_bibtex(
    *,
    key: str,
    title: str,
    authors: list[str],
    year: int | None,
    venue: str | None,
    doi: str | None,
    url: str | None,
) -> str:
    """Generate a valid BibTeX entry from already-fetched metadata.

    Used by JSON-first sources (Crossref, OpenAlex, Semantic Scholar) that do
    not expose a native BibTeX endpoint without one extra request per paper.
    """
    fields = [f"  title = {{{title}}}"]
    if authors:
        fields.append(f"  author = {{{' and '.join(authors)}}}")
    if year is not None:
        fields.append(f"  year = {{{year}}}")
    if venue:
        fields.append(f"  journal = {{{venue}}}")
    if doi:
        fields.append(f"  doi = {{{doi}}}")
    if url:
        fields.append(f"  url = {{{url}}}")
    body = ",\n".join(fields)
    return f"@article{{{key},\n{body}\n}}"


def sanitize_query(query: str, max_length: int = 300) -> str:
    """Collapse whitespace and cap length; providers reject long queries."""
    cleaned = " ".join(query.split())
    return cleaned[:max_length]


def quote_doi_for_url(doi: str) -> str:
    return quote(doi, safe="")
