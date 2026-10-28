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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

import httpx

from paper_agent.domain.errors import FullTextParseError, ProviderUnavailableError

MINERU_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
#: Bump whenever text-to-Markdown or heading rules change; cached Markdown
#: produced by an older converter is then re-generated.
CONVERTER_VERSION = "group3-md-v3"
MAX_RESULT_ZIP_BYTES = 50_000_000
MAX_RESULT_BYTES = 100_000_000
MAX_IMAGE_BYTES = 10_000_000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
IMAGE_LINK = re.compile(r"!\[[^\]\n]*\]\(\s*(?P<url><[^>\n]+>|[^\s)]+)(?:\s+\"[^\"\n]*\")?\s*\)")


def safe_asset_name(value: str) -> str | None:
    """仅接受相对图片路径；不允许 ZIP 路径穿越、协议和 Windows 路径。"""
    value = unquote(value.strip().strip("<>"))
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc or "\\" in value or "\x00" in value or ":" in value:
        return None
    path = PurePosixPath(parsed.path)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return path.as_posix()


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
    assets: dict[str, bytes] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


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
    garbage_count = text.count("\ufffd") + len(re.findall(r"[\x00-\x08\x0b-\x1f]", text))
    garbage = garbage_count / max(1, len(text))
    score = 0.6 * coverage + 0.3 * structure + 0.1 * (1 - garbage)
    return round(max(0.0, min(1.0, score)), 4)


def _text_to_markdown(text: str, fallback_title: str) -> str:
    """合并 PDF 硬换行，保留段落、章节、列表、表格、代码和已有公式。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    blocks: list[str] = []
    paragraph = ""
    title_written = False
    protected: str | None = None

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(paragraph)
            paragraph = ""

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if protected:
            blocks.append(raw_line)
            if (
                (protected in {"```", "~~~"} and line.startswith(protected))
                or (protected == "$$" and line.endswith("$$"))
                or (protected == r"\]" and line.endswith(r"\]"))
            ):
                protected = None
            continue
        if line.startswith(("```", "~~~", "$$", r"\[")):
            flush()
            blocks.append(raw_line)
            if line.startswith(("```", "~~~")):
                protected = line[:3]
            elif line == "$$" or (line.startswith("$$") and not line[2:].endswith("$$")):
                protected = "$$"
            elif line.startswith(r"\[") and not line.endswith(r"\]"):
                protected = r"\]"
            continue
        if not line:
            flush()
            if blocks and blocks[-1] != "":
                blocks.append("")
            continue
        if _canonical_section(line) is not None or line.startswith("#"):
            flush()
            blocks.extend(["", line if line.startswith("#") else f"## {line}", ""])
            title_written = True
            continue
        if not title_written:
            if len(line) <= 160 and not re.search(r"[.!?。！？]$", line):
                blocks.extend([f"# {line}", ""])
                title_written = True
                continue
            blocks.extend([f"# {fallback_title}", ""])
            title_written = True
        if re.match(r"^(?:[-*+•]\s|\d+[.)]\s|\[\d+\]\s|>|!\[|\|)", line):
            flush()
            blocks.append(line)
            continue
        if (
            paragraph.endswith("-")
            and re.search(r"[A-Za-z]-$", paragraph)
            and re.match(r"[a-z]", line)
        ):
            paragraph = paragraph[:-1] + line
        elif paragraph:
            # 中文排版硬换行之间不插入英文空格。
            sep = (
                ""
                if re.search(r"[\u3400-\u9fff]$", paragraph) and re.match(r"[\u3400-\u9fff]", line)
                else " "
            )
            paragraph += sep + line
        else:
            paragraph = line
    flush()
    return "\n".join(blocks).strip() + "\n"


def _normalize_math(markdown: str) -> str:
    """统一已有 LaTeX 分隔符供 Markdown 渲染；不猜测或编造公式，代码块保持原样。"""
    parts = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)", markdown)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", r"\n$$\n\1\n$$\n", parts[index])
        parts[index] = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", parts[index])
    return "".join(parts).strip() + "\n"


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
                    'pypdf is not installed; run `pip install -e ".[processing]"`'
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
            assets: dict[str, bytes] = {}
            warnings = [
                (
                    "pypdf 仅提取文本层和嵌入位图，无法可靠识别公式、扫描正文或矢量图；"
                    "需要公式/版面识别时请选择 MinerU，并配置 MINERU_API_KEY。"
                )
            ]
            total_bytes = 0
            for page_index, page in enumerate(reader.pages, 1):
                try:
                    for image_index, image in enumerate(page.images, 1):
                        suffix = Path(image.name).suffix.lower()
                        data = image.data
                        if suffix not in IMAGE_SUFFIXES:
                            buffer = io.BytesIO()
                            image.image.save(buffer, format="PNG")
                            data, suffix = buffer.getvalue(), ".png"
                        if (
                            len(data) > MAX_IMAGE_BYTES
                            or total_bytes + len(data) > MAX_RESULT_BYTES
                        ):
                            warnings.append(f"第 {page_index} 页图片超出资源大小上限，已跳过。")
                            continue
                        name = f"images/page-{page_index:04d}-{image_index:03d}{suffix}"
                        assets[name] = data
                        total_bytes += len(data)
                except Exception:  # 图片失败不能丢弃已成功提取的正文。 # noqa: BLE001
                    warnings.append(f"第 {page_index} 页部分嵌入图片无法提取，请参照原始 PDF。")
        except FullTextParseError:
            raise
        except Exception as exc:  # pypdf raises assorted errors on broken files
            raise FullTextParseError(f"pypdf failed to read {pdf_path.name}: {exc}") from exc
        return ExtractedDocument(
            text="\n\n".join(pages), page_count=page_count, assets=assets, warnings=warnings
        )


class _TransientMineruError(ProviderUnavailableError):
    """网络抖动、超时或服务端 5xx：换一次重试往往就能成功。"""


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


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
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 3.0,
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
        self._retry_attempts = max(1, retry_attempts)
        self._retry_backoff = max(0.0, retry_backoff_seconds)
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
        attempts = self._retry_attempts
        last: _TransientMineruError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._extract_once(pdf_path)
            except _TransientMineruError as exc:
                last = exc
                if attempt >= attempts:
                    break
                # 批量解析失败的绝大多数是网络抖动或服务端 5xx，重试一次往往就成；
                # 鉴权、额度、单篇不支持这类错误不会走到这里。
                await asyncio.sleep(self._retry_backoff * attempt)
        raise ProviderUnavailableError(f"MinerU 连续 {attempts} 次调用失败：{last}") from last

    async def _extract_once(self, pdf_path: Path) -> ExtractedDocument:
        timeout = httpx.Timeout(self._timeout, connect=min(15.0, self._timeout))
        # 凭证仅发往 MinerU API，不能随预签名上传地址或结果 CDN 请求泄露。
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            batch_id, upload_url = await self._request_upload(client, pdf_path)
            await self._upload(client, upload_url, pdf_path)
            zip_url = await self._poll_batch(client, batch_id)
            document = await self._download_markdown(client, zip_url)
        document.page_count = await self._safe_page_count(pdf_path)
        return document

    async def _request_upload(self, client: httpx.AsyncClient, pdf_path: Path) -> tuple[str, str]:
        payload = {
            "files": [
                {
                    "name": pdf_path.name,
                    "is_ocr": True,
                    "data_id": pdf_path.stem[:120],
                }
            ],
            "model_version": self._model_version,
            "enable_formula": True,
            "enable_table": True,
        }
        try:
            response = await client.post(
                f"{self._base_url}/file-urls/batch",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise _TransientMineruError(f"MinerU upload request failed: {exc}") from exc
        if response.status_code != 200:
            error_type = (
                _TransientMineruError
                if _is_retryable_status(response.status_code)
                else ProviderUnavailableError
            )
            raise error_type(f"MinerU returned HTTP {response.status_code} for upload request")
        try:
            body = response.json()
        except ValueError as exc:
            raise _TransientMineruError(f"Malformed MinerU upload response: {exc}") from exc
        if body.get("code") not in (0, None):
            raise ProviderUnavailableError(f"MinerU rejected upload request: {body.get('msg')}")
        try:
            batch_id = str(body["data"]["batch_id"])
            upload_url = str(body["data"]["file_urls"][0])
        except (KeyError, IndexError, TypeError) as exc:
            raise _TransientMineruError(f"Malformed MinerU upload response: {exc}") from exc
        return batch_id, upload_url

    async def _upload(self, client: httpx.AsyncClient, upload_url: str, pdf_path: Path) -> None:
        content = await asyncio.to_thread(pdf_path.read_bytes)
        try:
            # MinerU requires no Content-Type header on the presigned PUT.
            response = await client.put(upload_url, content=content)
        except httpx.HTTPError as exc:
            raise _TransientMineruError(f"MinerU file upload failed: {exc}") from exc
        if response.status_code != 200:
            error_type = (
                _TransientMineruError
                if _is_retryable_status(response.status_code)
                else ProviderUnavailableError
            )
            raise error_type(f"MinerU file upload returned HTTP {response.status_code}")

    async def _poll_batch(self, client: httpx.AsyncClient, batch_id: str) -> str:
        deadline = asyncio.get_running_loop().time() + self._poll_timeout
        while True:
            try:
                response = await client.get(
                    f"{self._base_url}/extract-results/batch/{batch_id}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            except httpx.HTTPError as exc:
                raise _TransientMineruError(f"MinerU polling failed: {exc}") from exc
            if response.status_code != 200:
                error_type = (
                    _TransientMineruError
                    if _is_retryable_status(response.status_code)
                    else ProviderUnavailableError
                )
                raise error_type(f"MinerU polling returned HTTP {response.status_code}")
            try:
                results = response.json()["data"]["extract_result"]
            except (ValueError, KeyError, TypeError) as exc:
                raise _TransientMineruError(f"Malformed MinerU poll response: {exc}") from exc
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
                raise _TransientMineruError(
                    f"MinerU polling timed out after {self._poll_timeout:.0f}s"
                )
            await asyncio.sleep(self._poll_interval)

    async def _download_markdown(
        self, client: httpx.AsyncClient, zip_url: str
    ) -> ExtractedDocument:
        content = bytearray()
        try:
            async with client.stream("GET", zip_url, follow_redirects=True) as response:
                if response.status_code != 200:
                    raise _TransientMineruError(
                        f"MinerU result download returned HTTP {response.status_code}"
                    )
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_RESULT_ZIP_BYTES:
                        raise FullTextParseError("MinerU 结果压缩包超过大小上限。")
        except httpx.HTTPError as exc:
            raise _TransientMineruError(f"MinerU result download failed: {exc}") from exc
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 1000 or sum(e.file_size for e in entries) > MAX_RESULT_BYTES:
                    raise FullTextParseError("MinerU 结果解压大小或文件数量超过上限。")
                target = next(
                    (e for e in entries if PurePosixPath(e.filename).name.lower() == "full.md"),
                    None,
                )
                if target is None:
                    raise FullTextParseError("MinerU result zip has no full.md")
                base = PurePosixPath(target.filename).parent
                if base.is_absolute() or ".." in base.parts or "\\" in target.filename:
                    raise FullTextParseError("MinerU Markdown 路径不安全。")
                if target.file_size > 20_000_000:
                    raise FullTextParseError("MinerU Markdown 超过大小上限。")
                text = archive.read(target).decode("utf-8", errors="replace")
                assets = {}
                warnings = []
                for entry in entries:
                    path = PurePosixPath(entry.filename)
                    if entry.is_dir() or not path.is_relative_to(base):
                        continue
                    name = safe_asset_name(path.relative_to(base).as_posix())
                    if name is None:
                        continue
                    if entry.file_size > MAX_IMAGE_BYTES:
                        warnings.append(f"图片 {name} 超过大小上限，已跳过。")
                        continue
                    assets[name] = archive.read(entry)
                return ExtractedDocument(
                    text=text, markdown_ready=True, assets=assets, warnings=warnings
                )
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise _TransientMineruError(f"MinerU result is not a valid zip: {exc}") from exc

    async def _safe_page_count(self, pdf_path: Path) -> int | None:
        def count_pages() -> int | None:
            try:
                import pypdf

                # 这里只需页数，不能再次提取整篇文本和图片，拖慢已完成的 MinerU 解析。
                return len(pypdf.PdfReader(str(pdf_path)).pages)
            except Exception:  # 本地页数读取失败不影响远程解析结果。 # noqa: BLE001
                return None

        return await asyncio.to_thread(count_pages)


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

    def cache_is_current(self, pdf_path: Path, markdown_path: Path) -> bool:
        """供全文门面检查缓存，切换后端或丢失图片后必须重新解析。"""
        return (
            self._read_cache(
                markdown_path, markdown_path.with_suffix(".parse.json"), _sha256_file(pdf_path)
            )
            is not None
        )

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
        warnings = list(document.warnings)
        asset_prefix = f"{stem}_assets"
        asset_hashes = {}
        total_bytes = 0
        for name, data in document.assets.items():
            safe_name = safe_asset_name(name)
            if safe_name is None or len(data) > MAX_IMAGE_BYTES:
                warnings.append("部分图片路径不安全或文件过大，已跳过。")
                continue
            total_bytes += len(data)
            if total_bytes > MAX_RESULT_BYTES:
                raise FullTextParseError("图片资源总大小超过上限。")
            relative = f"{asset_prefix}/{safe_name}"
            path = markdown_dir / relative
            if not path.resolve().is_relative_to(markdown_dir.resolve()):
                raise FullTextParseError("图片保存路径越界。")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_atomically(path, data)
            asset_hashes[relative] = hashlib.sha256(data).hexdigest()
        if not document.markdown_ready and asset_hashes:
            markdown += "\n## 嵌入图片（按页提取）\n\n"
            markdown += "\n\n".join(f"![{name}]({name})" for name in document.assets) + "\n"

        def rewrite_image(match: re.Match) -> str:
            name = safe_asset_name(match.group("url"))
            relative = f"{asset_prefix}/{name}" if name else ""
            if relative not in asset_hashes:
                warnings.append("存在未保存或不安全的图片引用，请参照原始 PDF。")
                return "[图片不可用，请参照原始 PDF]"
            start, end = match.span("url")
            return (
                match.group(0)[: start - match.start()]
                + quote(relative, safe="/")
                + match.group(0)[end - match.start() :]
            )

        markdown = _normalize_math(IMAGE_LINK.sub(rewrite_image, markdown))
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
            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "assets": asset_hashes,
            "warnings": list(dict.fromkeys(warnings)),
        }
        self._write_atomically(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )
        return markdown_path, quality

    def _read_cache(self, markdown_path: Path, meta_path: Path, pdf_sha256: str) -> float | None:
        if not (markdown_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            if meta.get("pdf_sha256") != pdf_sha256:
                return None
            if meta.get("converter_version") != CONVERTER_VERSION:
                return None
            if (
                meta.get("parser_name") != self.parser_name
                or meta.get("parser_version") != self.parser_version
            ):
                return None
            if meta.get("markdown_sha256") != _sha256_file(markdown_path):
                return None
            for name, digest in meta.get("assets", {}).items():
                safe_name = safe_asset_name(name)
                if safe_name is None:
                    return None
                path = markdown_path.parent / safe_name
                if not path.resolve().is_relative_to(markdown_path.parent.resolve()):
                    return None
                if _sha256_file(path) != digest:
                    return None
            return max(0.0, min(1.0, float(meta["quality"])))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    @staticmethod
    def _write_atomically(path: Path, data: bytes) -> None:
        # 同一 PDF 可能被多个任务同时处理，临时文件不能共享固定名称。
        tmp = path.with_name(path.name + f".{uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
