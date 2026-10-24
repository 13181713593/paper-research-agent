"""Group 3: convert a local PDF into Markdown with a quality score.

Two extractor backends are supported:

- ``pypdf``: local, pure-Python text-layer extraction; no network, no models.
  Scanned PDFs without a text layer fail with ``FullTextParseError``;
- ``mineru``: the MinerU precision parsing API (https://mineru.net, v4 batch
  file upload flow), which handles OCR, formulas and tables. Requires an API
  token (``MINERU_API_KEY``).

``backend="auto"`` prefers MinerU whenever a token is configured and falls
back to local pypdf otherwise. Results are cached next to the PDF (sidecar
``.parse.json`` keyed by the PDF content hash), so repeated runs never
re-parse the same document.

``split_sections`` and ``score_parse_quality`` are shared with the summarizer
so section-aware extraction stays consistent across Group 3 components.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import zipfile
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from paper_agent.domain.errors import FullTextParseError, ProviderUnavailableError

MINERU_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
#: Bump whenever text-to-Markdown or heading rules change; cached Markdown
#: produced by an older converter is then re-generated.
CONVERTER_VERSION = "group3-md-v2"

#: Canonical section names recognised in extracted text. The aliases cover
#: common headings with or without numbering and markdown prefixes.
_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "preliminaries": "preliminaries",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "model architecture": "method",
    "proposed method": "method",
    "proposed approach": "method",
    "experiment": "experiments",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experiment settings": "experiments",
    "experimental settings": "experiments",
    "evaluation": "experiments",
    "results": "experiments",
    "results and discussion": "experiments",
    "experimental results": "experiments",
    "experiments and results": "experiments",
    "analysis": "analysis",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
    "future work": "future_work",
    "references": "references",
    "appendix": "appendix",
}

_STRUCTURE_SECTIONS = {
    "abstract",
    "introduction",
    "method",
    "experiments",
    "conclusion",
}
# Markdown hashes and section numbering may be stacked ("## 1 Introduction").
_HEADING_PREFIX = re.compile(r"^(?:#{1,6}\s*)?(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?")
# Small-caps headings extract as "I NTRODUCTION"; rejoin the split capital.
_SMALL_CAPS_SPLIT = re.compile(r"\b([A-Z]) (?=[A-Z]{2,})")


@dataclass(slots=True)
class ExtractedDocument:
    """Raw extraction result handed over from an extractor to the parser."""

    text: str
    page_count: int | None = None
    markdown_ready: bool = False


class TextExtractor(Protocol):
    """Pluggable backend: turns a local PDF into text (or ready Markdown)."""

    name: str
    version: str

    async def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text from ``pdf_path`` or raise ``FullTextParseError``."""
        ...


def _canonical_section(line: str) -> str | None:
    """Map a heading-looking line to a canonical section name or ``None``."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    stripped = _HEADING_PREFIX.sub("", stripped).strip(" .:：\t")
    stripped = _SMALL_CAPS_SPLIT.sub(r"\1", stripped)
    key = re.sub(r"\s+", " ", stripped.lower())
    return _SECTION_ALIASES.get(key)


def split_sections(text: str) -> dict[str, str]:
    """Split text into ``{canonical_section: content}`` blocks.

    Content before the first recognised heading lands in ``"body"``. Repeated
    headings are merged into a single block; heading lines themselves are not
    part of the content. Used by the extractive summarizer and by the
    structure part of the quality score.
    """
    sections: dict[str, list[str]] = {}
    current = "body"
    for raw_line in text.splitlines():
        name = _canonical_section(raw_line)
        if name is not None:
            current = name
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def score_parse_quality(text: str, page_count: int | None = None) -> float:
    """Heuristic Markdown quality score in ``[0, 1]``.

    Combines text coverage (characters per page when the page count is known),
    structural completeness (recognised section headings) and a garbage ratio
    (replacement/control characters). Formula version: ``group3-quality-v1``.
    """
    compact = re.sub(r"\s+", "", text)
    characters = len(compact)
    if characters == 0:
        return 0.0
    if page_count and page_count > 0:
        coverage = min(1.0, characters / (page_count * 900))
    else:
        coverage = min(1.0, characters / 6000)
    found = {name for name in split_sections(text) if name in _STRUCTURE_SECTIONS}
    structure = min(1.0, len(found) / 4)
    garbage_count = text.count("\ufffd") + len(
        re.findall(r"[\x00-\x08\x0b-\x1f]", text)
    )
    garbage = garbage_count / max(1, len(text))
    score = 0.6 * coverage + 0.3 * structure + 0.1 * (1 - garbage)
    return round(max(0.0, min(1.0, score)), 4)


def _text_to_markdown(text: str, fallback_title: str) -> str:
    """Light formatting of raw extracted text into readable Markdown."""
    normalized = text.replace("\r\n", "\n").replace("\u00ad", "")
    # Re-join words hyphenated across line breaks ("docu-\nment" -> "document").
    normalized = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", normalized)
    lines_out: list[str] = []
    title_written = False
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            if lines_out and lines_out[-1] != "":
                lines_out.append("")
            continue
        name = _canonical_section(line)
        if name is not None:
            lines_out.append(f"## {line}")
            title_written = True
            continue
        if not title_written and len(line) <= 160:
            lines_out.extend([f"# {line}", ""])
            title_written = True
            continue
        lines_out.append(line)
    if not title_written:
        lines_out.insert(0, f"# {fallback_title}")
    markdown = "\n".join(lines_out).strip()
    return markdown + "\n"


def _extract_in_worker(pdf_path: Path) -> ExtractedDocument:
    """进程池入口保持在模块顶层，兼容 macOS/Windows 的 spawn。"""
    return PypdfExtractor()._extract_sync(pdf_path)


class PypdfExtractor:
    """Local text-layer extraction through the optional ``pypdf`` package."""

    name = "pypdf"

    def __init__(self, executor: Executor | None = None) -> None:
        self._module = None
        self._executor = executor

    def _import(self):
        if self._module is None:
            try:
                import pypdf
            except ImportError as exc:  # pragma: no cover - depends on env
                raise FullTextParseError(
                    "pypdf is not installed; run `pip install -e \".[processing]\"`"
                ) from exc
            self._module = pypdf
        return self._module

    @property
    def version(self) -> str:
        try:
            return str(getattr(self._import(), "__version__", "unknown"))
        except FullTextParseError:
            return "unavailable"

    async def extract(self, pdf_path: Path) -> ExtractedDocument:
        if self._executor is not None:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, _extract_in_worker, pdf_path
            )
        return await asyncio.to_thread(self._extract_sync, pdf_path)

    def _extract_sync(self, pdf_path: Path) -> ExtractedDocument:
        pypdf = self._import()
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            page_count = len(reader.pages)
        except FullTextParseError:
            raise
        except Exception as exc:  # pypdf raises assorted errors on broken files
            raise FullTextParseError(
                f"pypdf failed to read {pdf_path.name}: {exc}"
            ) from exc
        return ExtractedDocument(
            text="\n\n".join(pages), page_count=page_count, markdown_ready=False
        )


class MineruExtractor:
    """MinerU precision parsing API (v4 batch upload flow).

    Uploads the PDF, polls the batch result endpoint, downloads the result
    zip and reads ``full.md``. Handles OCR, formulas and tables server-side.
    """

    name = "mineru"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = MINERU_DEFAULT_BASE_URL,
        model_version: str = "vlm",
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 5.0,
        poll_timeout_seconds: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``None`` means "read the environment"; "" explicitly disables the
        # remote backend (deterministic for tests and offline runs).
        if api_key is None:
            api_key = os.getenv("MINERU_API_KEY") or ""
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model_version = model_version
        self._timeout = timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._transport = transport

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def version(self) -> str:
        return f"api-v4:{self._model_version}"

    async def extract(self, pdf_path: Path) -> ExtractedDocument:
        if not self.available:
            raise FullTextParseError("MINERU_API_KEY is not configured")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        timeout = httpx.Timeout(self._timeout, connect=min(15.0, self._timeout))
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, transport=self._transport
        ) as client:
            batch_id, upload_url = await self._request_upload(client, pdf_path)
            await self._upload(client, upload_url, pdf_path)
            zip_url = await self._poll_batch(client, batch_id)
            markdown = await self._download_markdown(client, zip_url)
        page_count = await self._safe_page_count(pdf_path)
        return ExtractedDocument(
            text=markdown, page_count=page_count, markdown_ready=True
        )

    async def _request_upload(
        self, client: httpx.AsyncClient, pdf_path: Path
    ) -> tuple[str, str]:
        payload = {
            "files": [
                {
                    "name": pdf_path.name,
                    "is_ocr": True,
                    "data_id": pdf_path.stem[:120],
                }
            ],
            "model_version": self._model_version,
        }
        try:
            response = await client.post(f"{self._base_url}/file-urls/batch", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"MinerU upload request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"MinerU returned HTTP {response.status_code} for upload request"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(f"Malformed MinerU upload response: {exc}") from exc
        if body.get("code") not in (0, None):
            raise ProviderUnavailableError(f"MinerU rejected upload request: {body.get('msg')}")
        try:
            batch_id = str(body["data"]["batch_id"])
            upload_url = str(body["data"]["file_urls"][0])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(f"Malformed MinerU upload response: {exc}") from exc
        return batch_id, upload_url

    async def _upload(
        self, client: httpx.AsyncClient, upload_url: str, pdf_path: Path
    ) -> None:
        content = await asyncio.to_thread(pdf_path.read_bytes)
        try:
            # MinerU requires no Content-Type header on the presigned PUT.
            response = await client.put(upload_url, content=content)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"MinerU file upload failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"MinerU file upload returned HTTP {response.status_code}"
            )

    async def _poll_batch(self, client: httpx.AsyncClient, batch_id: str) -> str:
        deadline = asyncio.get_running_loop().time() + self._poll_timeout
        while True:
            try:
                response = await client.get(
                    f"{self._base_url}/extract-results/batch/{batch_id}"
                )
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(f"MinerU polling failed: {exc}") from exc
            if response.status_code != 200:
                raise ProviderUnavailableError(
                    f"MinerU polling returned HTTP {response.status_code}"
                )
            try:
                results = response.json()["data"]["extract_result"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderUnavailableError(f"Malformed MinerU poll response: {exc}") from exc
            if isinstance(results, dict):
                results = [results]
            for entry in results:
                state = entry.get("state")
                if state == "done" and entry.get("full_zip_url"):
                    return str(entry["full_zip_url"])
                if state == "failed":
                    raise FullTextParseError(
                        f"MinerU parsing failed: {entry.get('err_msg') or 'unknown error'}"
                    )
            if asyncio.get_running_loop().time() >= deadline:
                raise ProviderUnavailableError(
                    f"MinerU polling timed out after {self._poll_timeout:.0f}s"
                )
            await asyncio.sleep(self._poll_interval)

    async def _download_markdown(self, client: httpx.AsyncClient, zip_url: str) -> str:
        try:
            response = await client.get(zip_url)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"MinerU result download failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"MinerU result download returned HTTP {response.status_code}"
            )
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = archive.namelist()
                target = next(
                    (name for name in names if name.lower().endswith("full.md")),
                    None,
                )
                if target is None:
                    raise FullTextParseError(
                        f"MinerU result zip has no full.md (files={names[:5]})"
                    )
                return archive.read(target).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise ProviderUnavailableError(f"MinerU result is not a valid zip: {exc}") from exc

    async def _safe_page_count(self, pdf_path: Path) -> int | None:
        try:
            document = await PypdfExtractor().extract(pdf_path)
            return document.page_count
        except FullTextParseError:
            return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PdfMarkdownParser:
    """``MarkdownParser`` implementation with extractor selection + cache."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        extractor: TextExtractor | None = None,
        mineru_api_key: str | None = None,
        mineru_model_version: str = "vlm",
        transport: httpx.AsyncBaseTransport | None = None,
        min_characters: int = 50,
        executor: Executor | None = None,
    ) -> None:
        if extractor is not None:
            self._extractor = extractor
        elif backend not in {"auto", "pypdf", "mineru"}:
            raise ValueError(f"unknown markdown backend '{backend}'")
        elif backend == "pypdf":
            self._extractor = PypdfExtractor(executor=executor)
        elif backend == "mineru":
            self._extractor = MineruExtractor(
                api_key=mineru_api_key,
                model_version=mineru_model_version,
                transport=transport,
            )
        else:  # auto
            mineru = MineruExtractor(
                api_key=mineru_api_key,
                model_version=mineru_model_version,
                transport=transport,
            )
            self._extractor = mineru if mineru.available else PypdfExtractor(executor=executor)
        self._min_characters = min_characters

    @property
    def parser_name(self) -> str:
        return self._extractor.name

    @property
    def parser_version(self) -> str:
        return self._extractor.version

    async def parse(self, pdf_path: Path, destination_dir: Path) -> tuple[Path, float]:
        """Convert ``pdf_path`` to Markdown returning ``(path, quality)``.

        Raises:
            FullTextParseError: extraction produced no usable text (for
                example a scanned PDF without a text layer under pypdf).
            ProviderUnavailableError: the remote MinerU backend is unreachable.
        """
        markdown_dir = Path(destination_dir)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem
        markdown_path = markdown_dir / f"{stem}.md"
        meta_path = markdown_dir / f"{stem}.parse.json"
        pdf_sha256 = await asyncio.to_thread(_sha256_file, pdf_path)
        cached = self._read_cache(markdown_path, meta_path, pdf_sha256)
        if cached is not None:
            return markdown_path, cached
        document = await self._extractor.extract(pdf_path)
        text = document.text or ""
        compact = re.sub(r"\s+", "", text)
        if len(compact) < self._min_characters:
            raise FullTextParseError(
                f"no usable text extracted from {pdf_path.name} "
                f"(characters={len(compact)}, extractor={self.parser_name})"
            )
        if document.markdown_ready:
            markdown = text.strip() + "\n"
        else:
            markdown = _text_to_markdown(text, fallback_title=stem)
        quality = score_parse_quality(text, document.page_count)
        self._write_atomically(markdown_path, markdown.encode("utf-8"))
        meta = {
            "pdf_sha256": pdf_sha256,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "quality": quality,
            "parsed_at": datetime.now(UTC).isoformat(),
            "formula_version": "group3-quality-v1",
            "converter_version": CONVERTER_VERSION,
        }
        self._write_atomically(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )
        return markdown_path, quality

    @staticmethod
    def _read_cache(
        markdown_path: Path, meta_path: Path, pdf_sha256: str
    ) -> float | None:
        if not (markdown_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            if meta.get("pdf_sha256") != pdf_sha256:
                return None
            if meta.get("converter_version") != CONVERTER_VERSION:
                return None
            return max(0.0, min(1.0, float(meta["quality"])))
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _write_atomically(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
