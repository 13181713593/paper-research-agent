"""Group 3: ``FullTextProcessor`` facade combining resolver, downloader and parser.

The pipeline calls this component concurrently for every paper in the
full-text candidate pool. The implementation keeps no shared mutable state
beyond the on-disk caches owned by the downloader/parser, so bounded
concurrent calls are safe.

Contract behaviour (see ``docs/API_CONTRACTS.md``):

- input/output use the same paper identity; only ``fulltext`` fields are
  updated, in place, so failure observations survive the orchestrator's
  abstract-fallback path;
- typed failures propagate so the orchestrator can degrade to abstract
  fallback or exclude the paper; before re-raising, the failure state
  (``access_status`` / ``parse_error``) is recorded on the paper.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from paper_agent.domain.models import AccessStatus, Paper
from paper_agent.ports.contracts import MarkdownParser, OpenAccessResolver, PdfDownloader


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PdfFullTextProcessor:
    """Compose OA resolution -> download -> Markdown parsing for one paper."""

    def __init__(
        self,
        *,
        resolver: OpenAccessResolver,
        downloader: PdfDownloader,
        parser: MarkdownParser,
        work_dir: Path,
    ) -> None:
        self._resolver = resolver
        self._downloader = downloader
        self._parser = parser
        self._pdf_dir = Path(work_dir) / "pdfs"
        self._markdown_dir = Path(work_dir) / "markdown"

    async def process(self, paper: Paper) -> Paper:
        try:
            async with asyncio.timeout(90):
                return await self._process(paper)
        except TimeoutError:
            paper.fulltext.parse_error = "全文定位、下载和解析超过90秒预算，保留元数据。"
            raise

    async def _process(self, paper: Paper) -> Paper:
        """Process one paper, updating its ``fulltext`` fields in place.

        Short-circuits when the paper already carries an existing Markdown
        file, so repeated runs do not re-download or re-parse.

        Raises:
            NoOpenAccessCopyError: no lawful OA copy (paywall, closed,
                removed, or no URL evidence).
            ProviderUnavailableError: transport failure or invalid payload.
            FullTextParseError: download succeeded but no usable text was
                extracted; ``parse_error`` records the reason on the paper.
        """
        existing_markdown = paper.fulltext.markdown_path
        if existing_markdown and Path(existing_markdown).exists():
            return paper
        try:
            resolved = await self._resolver.resolve(paper)
        except Exception as exc:
            paper.fulltext.parse_error = f"{type(exc).__name__}: {exc}"
            raise
        urls = list(
            dict.fromkeys(
                [
                    str(resolved.fulltext.pdf_url),
                    *(str(u) for u in resolved.fulltext.candidate_urls),
                ]
            )
        )
        urls = [url for url in urls if url.startswith(("https://", "http://"))]
        discovered = False
        pdf_path = None
        last_error = None
        index = 0
        while index < min(len(urls), 4):
            url = urls[index]
            index += 1
            resolved.fulltext.pdf_url = url
            try:
                pdf_path = await self._downloader.download(resolved, self._pdf_dir)
                paper.fulltext.attempts.append(
                    {"stage": "download", "url": url, "status": "succeeded"}
                )
                break
            except Exception as exc:  # noqa: BLE001 - retain failure and try another lawful copy
                last_error = exc
                paper.fulltext.attempts.append(
                    {"stage": "download", "url": url, "status": "failed", "error": str(exc)[:400]}
                )
                if index == len(urls) and not discovered and hasattr(self._resolver, "discover"):
                    discovered = True
                    await self._resolver.discover(paper)
                    urls.extend(str(u) for u in paper.fulltext.candidate_urls if str(u) not in urls)
        if pdf_path is None:
            paper.fulltext.access_status = AccessStatus.UNAVAILABLE
            paper.fulltext.parse_error = f"{type(last_error).__name__}: {last_error}"
            if last_error:
                raise last_error
            raise ValueError("No usable PDF candidate URL")
        pdf_path = Path(pdf_path)
        try:
            markdown_path, quality = await self._parser.parse(pdf_path, self._markdown_dir)
        except Exception as exc:
            paper.fulltext.parse_error = f"{type(exc).__name__}: {exc}"
            raise
        paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
        paper.fulltext.local_pdf_path = str(pdf_path)
        paper.fulltext.markdown_path = str(markdown_path)
        paper.fulltext.sha256 = await asyncio.to_thread(_sha256_file, pdf_path)
        paper.fulltext.downloaded_at = datetime.now(UTC)
        parser_name = getattr(self._parser, "parser_name", None)
        parser_version = getattr(self._parser, "parser_version", None)
        if parser_name:
            paper.fulltext.parser_name = parser_name
        if parser_version:
            paper.fulltext.parser_version = parser_version
        paper.fulltext.parse_quality = quality
        paper.fulltext.parse_error = None
        return paper
