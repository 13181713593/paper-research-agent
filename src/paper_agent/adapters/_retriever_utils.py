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

import time
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import requests


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


_RETRY_AFTER_CAP_SECONDS = 120.0


def _retry_after(response: Any, fallback: float) -> float:
    """Respect the provider's Retry-After header (seconds or HTTP-date).

    Falls back to the caller's exponential backoff when absent/unparsable and
    caps the wait so one stubborn provider cannot stall the whole pipeline.
    """
    headers = getattr(response, "headers", None) or {}
    getter = getattr(headers, "get", None)
    if getter is None:
        return fallback
    value = getter("Retry-After")
    if not value:
        return fallback
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
        except (TypeError, ValueError):
            return fallback
        seconds = (when - datetime.now(UTC)).total_seconds()
    return max(0.0, min(seconds, _RETRY_AFTER_CAP_SECONDS))


def http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    max_retries: int = 3,
    initial_backoff: float = 0.5,
) -> Any:
    """Blocking GET returning parsed JSON with retry/backoff.

    Retries 429 and 5xx with exponential backoff (default 0.5s, 1s, 2s). A 429
    honors the provider's ``Retry-After`` header when present. A 429 that
    persists after all retries raises ``ProviderRateLimitedError``; connection
    problems raise ``ProviderUnavailableError``.
    """
    backoff = initial_backoff
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.Timeout as exc:
            raise ProviderTimeoutError(f"{url} timed out after {timeout}s") from exc
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise ProviderUnavailableError(f"{url} unreachable: {exc}") from exc

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderResponseError(f"{url} returned invalid JSON") from exc

        if response.status_code == 429:
            if attempt < max_retries:
                time.sleep(_retry_after(response, fallback=backoff))
                backoff *= 2
                continue
            raise ProviderRateLimitedError(f"{url} rate limited after {max_retries} retries")

        if 500 <= response.status_code < 600:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise ProviderUnavailableError(f"{url} server error {response.status_code}")

        # 4xx other than 429: not retryable, fail loudly. A 404 on these
        # search endpoints means the URL/path is wrong or a proxy intercepted
        # the request — silently returning "no results" would hide the real
        # failure from the pipeline's per-source warnings.
        raise ProviderResponseError(f"{url} returned HTTP {response.status_code}: {response.text[:200]}")


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
            doi = doi[len(prefix):]
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
