"""Group 3: download one PDF with hard safety limits and a disk cache.

Safety envelope enforced here (see ``docs/INTEGRATION_CHECKLIST.md``):

- only ``http``/``https`` URLs and a bounded redirect chain;
- download timeout, maximum byte budget checked against both the
  ``Content-Length`` header *and* the streamed body;
- rejection of obvious non-PDF payloads by content type plus the ``%PDF-``
  magic bytes in the first KiB;
- cache keyed by a stable URL/DOI/arXiv id so repeated runs neither
  re-download nor re-store the same document; the recorded sha256 doubles as
  a content hash for auditability.

Failures raise typed errors: ``NoOpenAccessCopyError`` when the copy is
unavailable or forbidden, ``ProviderUnavailableError`` for transport errors
and invalid payloads so the orchestrator can degrade to abstract fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from paper_agent.domain.errors import NoOpenAccessCopyError, ProviderUnavailableError
from paper_agent.domain.models import Paper

_REDIRECT_CODES = {301, 302, 303, 307, 308}
_FORBIDDEN_CODES = {401, 402, 403, 404, 410}


class _FetchError(ProviderUnavailableError):
    """Internal error that records whether a bounded retry could help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class HttpPdfDownloader:
    """``PdfDownloader`` implementation built on ``httpx``."""

    def __init__(
        self,
        *,
        user_agent: str = "paper-research-agent/0.2 (course project)",
        timeout_seconds: float = 60.0,
        max_bytes: int = 30_000_000,
        retry_limit: int = 2,
        max_redirects: int = 5,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._retry_limit = retry_limit
        self._max_redirects = max_redirects
        self._retry_backoff = retry_backoff_seconds
        self._transport = transport

    async def download(self, paper: Paper, destination_dir: Path) -> Path:
        """Download and validate the paper's PDF, returning the cached path.

        Raises:
            NoOpenAccessCopyError: no http(s) URL, or the server forbids /
                does not have the file.
            ProviderUnavailableError: transport failures, oversized bodies,
                non-PDF payloads, or a redirect chain exceeding the limit.
        """
        url = self._select_url(paper)
        if url is None:
            raise NoOpenAccessCopyError("paper has no http(s) PDF URL to download")
        pdf_dir = destination_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(paper, url)
        pdf_path = pdf_dir / f"{key}.pdf"
        meta_path = pdf_dir / f"{key}.json"
        cached = self._read_cache(pdf_path, meta_path)
        if cached is not None:
            return cached
        content, final_url = await self._fetch_with_retry(url)
        self._validate_magic(content, url)
        self._write_atomically(pdf_path, content)
        meta = {
            "url": url,
            "final_url": final_url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        self._write_atomically(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )
        return pdf_path

    # ------------------------------------------------------------------ cache

    @staticmethod
    def _cache_key(paper: Paper, url: str) -> str:
        identity = f"{url}|{paper.doi or ''}|{paper.arxiv_id or ''}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _read_cache(pdf_path: Path, meta_path: Path) -> Path | None:
        """Return the cached path when a previous download is intact."""
        if not (pdf_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            expected_bytes = int(meta.get("bytes", -1))
        except (OSError, ValueError):
            return None
        if expected_bytes >= 0 and pdf_path.stat().st_size == expected_bytes:
            return pdf_path
        return None

    @staticmethod
    def _write_atomically(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    # ------------------------------------------------------------------ fetch

    @staticmethod
    def _select_url(paper: Paper) -> str | None:
        candidates = []
        if paper.fulltext.pdf_url:
            candidates.append(str(paper.fulltext.pdf_url))
        if paper.open_access_url:
            candidates.append(str(paper.open_access_url))
        for candidate in candidates:
            if urlsplit(candidate).scheme in {"http", "https"}:
                return candidate
        return None

    async def _fetch_with_retry(self, url: str) -> tuple[bytes, str]:
        attempts = self._retry_limit + 1
        for attempt in range(attempts):
            try:
                return await self._fetch_once(url)
            except _FetchError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                await asyncio.sleep(self._retry_backoff * (2**attempt))
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _fetch_once(self, url: str) -> tuple[bytes, str]:
        timeout = httpx.Timeout(self._timeout, connect=min(15.0, self._timeout))
        headers = {"User-Agent": self._user_agent, "Accept": "application/pdf,*/*"}
        current = url
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, transport=self._transport
        ) as client:
            for _ in range(self._max_redirects + 1):
                request = client.build_request("GET", current)
                try:
                    response = await client.send(request, stream=True)
                except httpx.HTTPError as exc:
                    raise _FetchError(
                        f"network failure for {current}: {exc}", retryable=True
                    ) from exc
                try:
                    if response.status_code in _REDIRECT_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise _FetchError(
                                f"redirect without Location header for {current}",
                                retryable=False,
                            )
                        try:
                            current = str(request.url.join(location))
                        except ValueError as exc:
                            raise _FetchError(
                                f"invalid redirect target '{location}'", retryable=False
                            ) from exc
                        continue
                    return await self._read_response(current, response)
                finally:
                    await response.aclose()
            raise _FetchError(
                f"too many redirects (> {self._max_redirects}) for {url}",
                retryable=False,
            )

    async def _read_response(
        self, url: str, response: httpx.Response
    ) -> tuple[bytes, str]:
        status = response.status_code
        if status in _FORBIDDEN_CODES:
            raise NoOpenAccessCopyError(f"HTTP {status} for {url}")
        if status in {408, 429}:
            raise _FetchError(f"HTTP {status} (retryable) for {url}", retryable=True)
        if status >= 500:
            raise _FetchError(f"HTTP {status} for {url}", retryable=True)
        if status >= 400:
            raise _FetchError(f"HTTP {status} for {url}", retryable=False)
        content_type = (response.headers.get("content-type") or "").lower()
        if "text/" in content_type or "html" in content_type:
            raise _FetchError(
                f"server returned non-PDF content type '{content_type}' for {url}",
                retryable=False,
            )
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            raise _FetchError(
                f"PDF exceeds {self._max_bytes} bytes (Content-Length) for {url}",
                retryable=False,
            )
        buffer = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                if len(buffer) > self._max_bytes:
                    raise _FetchError(
                        f"PDF exceeds {self._max_bytes} bytes while streaming from {url}",
                        retryable=False,
                    )
        except httpx.HTTPError as exc:
            raise _FetchError(
                f"network failure while reading {url}: {exc}", retryable=True
            ) from exc
        if not buffer:
            raise _FetchError(f"empty response body from {url}", retryable=True)
        return bytes(buffer), str(response.url)

    def _validate_magic(self, content: bytes, url: str) -> None:
        if b"%PDF-" not in content[:1024]:
            raise ProviderUnavailableError(
                f"downloaded payload from {url} is not a PDF (magic check failed)"
            )
