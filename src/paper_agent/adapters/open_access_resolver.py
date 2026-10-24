"""Group 3: locate a lawful open-access copy of a paper.

Resolution order (cheapest and most reliable first):

1. an already verified ``fulltext.pdf_url``;
2. a PDF-looking ``open_access_url`` from provider metadata (other OA links
   are recorded as ``landing_page_url`` instead of being treated as PDFs);
3. a direct arXiv PDF link derived from ``arxiv_id``;
4. the Unpaywall REST API when a DOI and contact email are configured.

The resolver never bypasses paywalls. ``is_open_access`` stays provider
metadata; ``fulltext.access_status`` records what this resolver actually
observed. A typed ``NoOpenAccessCopyError`` is raised when no lawful copy is
available, so the pipeline can fall back to the abstract or exclude the paper.
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote, urlsplit

import httpx

from paper_agent.domain.errors import NoOpenAccessCopyError, ProviderUnavailableError
from paper_agent.domain.models import AccessStatus, Paper

UNPAYWALL_API = "https://api.unpaywall.org/v2"


def _http_url(value: str | None) -> str | None:
    """Return the value if it is a usable http(s) URL, otherwise ``None``."""
    if not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return value


def _looks_like_pdf(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


class UnpaywallResolver:
    """``OpenAccessResolver`` implementation built on direct links + Unpaywall."""

    def __init__(
        self,
        *,
        email: str | None = None,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``None`` means "read the environment"; "" explicitly disables the
        # Unpaywall lookup (deterministic for tests and offline runs).
        if email is None:
            email = os.getenv("UNPAYWALL_EMAIL") or ""
        self._email = email.strip()
        self._timeout = timeout_seconds
        self._transport = transport

    async def resolve(self, paper: Paper) -> Paper:
        """Update ``paper.fulltext`` in place and return the same object.

        In-place semantics matter after a failure: the orchestrator's
        abstract-fallback path keeps the original paper object, so the
        observed ``access_status`` (set before raising) stays visible.

        Raises:
            NoOpenAccessCopyError: no lawful copy could be located, or the
                Unpaywall record explicitly reports the DOI as closed.
            ProviderUnavailableError: the Unpaywall lookup failed (network,
                timeout, rate limit) and no direct link was available.
        """
        direct = self._direct_url(paper)
        if direct is not None:
            paper.fulltext.pdf_url = direct
            paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
            return paper
        if self._email and paper.doi:
            return await self._resolve_with_unpaywall(paper)
        paper.fulltext.access_status = (
            AccessStatus.CLOSED if paper.is_open_access is False else AccessStatus.UNAVAILABLE
        )
        raise NoOpenAccessCopyError(
            f"no lawful open-access copy found for '{paper.title[:80]}'"
        )

    def _direct_url(self, paper: Paper) -> str | None:
        """Pick the first usable direct PDF URL from metadata, or ``None``."""
        candidates: list[str] = []
        pdf_url = _http_url(str(paper.fulltext.pdf_url)) if paper.fulltext.pdf_url else None
        if pdf_url:
            candidates.append(pdf_url)
        oa_url = _http_url(str(paper.open_access_url)) if paper.open_access_url else None
        if oa_url:
            if _looks_like_pdf(oa_url):
                candidates.append(oa_url)
            elif not paper.fulltext.landing_page_url:
                # Provider OA link is a landing page; keep it for auditability.
                paper.fulltext.landing_page_url = oa_url
        if paper.arxiv_id:
            base_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
            if base_id:
                candidates.append(f"https://arxiv.org/pdf/{quote(base_id)}")
        return candidates[0] if candidates else None

    async def _resolve_with_unpaywall(self, paper: Paper) -> Paper:
        doi = (paper.doi or "").strip()
        url = f"{UNPAYWALL_API}/{quote(doi, safe='')}"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.get(url, params={"email": self._email})
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Unpaywall lookup failed for DOI {doi}: {exc}"
            ) from exc
        if response.status_code == 404:
            paper.fulltext.access_status = AccessStatus.UNAVAILABLE
            raise NoOpenAccessCopyError(f"Unpaywall has no record for DOI {doi}")
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Unpaywall returned HTTP {response.status_code} for DOI {doi}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(f"Unpaywall returned invalid JSON: {exc}") from exc
        candidate_locations = []
        if isinstance(data.get("best_oa_location"), dict):
            candidate_locations.append(data["best_oa_location"])
        candidate_locations.extend(
            loc for loc in data.get("oa_locations") or [] if isinstance(loc, dict)
        )
        pdf_url = None
        for location in candidate_locations:
            pdf_url = _http_url(location.get("url_for_pdf"))
            if pdf_url:
                break
        if pdf_url is None:
            pdf_url = next(
                (
                    _http_url(location.get("url"))
                    for location in candidate_locations
                    if _http_url(location.get("url"))
                ),
                None,
            )
        if not data.get("is_oa") or pdf_url is None:
            paper.fulltext.access_status = AccessStatus.CLOSED
            raise NoOpenAccessCopyError(f"Unpaywall reports no OA copy for DOI {doi}")
        paper.fulltext.pdf_url = pdf_url
        paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
        chosen = candidate_locations[0]
        license_name = chosen.get("license")
        if isinstance(license_name, str) and license_name:
            paper.fulltext.license = license_name
        version = chosen.get("version")
        if isinstance(version, str) and version:
            paper.fulltext.version = version
        if not paper.fulltext.landing_page_url:
            landing = _http_url(chosen.get("url_for_landing_page"))
            if landing:
                paper.fulltext.landing_page_url = landing
        return paper
