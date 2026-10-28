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
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path

from paper_agent.domain.errors import NoOpenAccessCopyError
from paper_agent.domain.models import AccessStatus, Paper
from paper_agent.ports.contracts import MarkdownParser, OpenAccessResolver, PdfDownloader


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PdfFullTextProcessor:
    """Compose OA resolution -> download -> Markdown parsing for one paper.

    定位/下载与解析分开计时：解析慢（云端解析排队）不应连带吞掉定位与下载
    的预算，反之定位失败也应快速结束而不是等满整体上限。
    """

    def __init__(
        self,
        *,
        resolver: OpenAccessResolver,
        downloader: PdfDownloader,
        parser: MarkdownParser,
        work_dir: Path,
        timeout_seconds: float | None = None,
        parse_timeout_seconds: float | None = None,
        download_concurrency: int = 8,
        parse_concurrency: int = 3,
        fallback_parser: MarkdownParser | None = None,
    ) -> None:
        self._resolver = resolver
        self._downloader = downloader
        self._parser = parser
        # 远端解析（MinerU）失败时退到本地 pypdf，避免整篇因解析服务不可用而失败。
        self._fallback_parser = fallback_parser
        self._pdf_dir = Path(work_dir) / "pdfs"
        self._markdown_dir = Path(work_dir) / "markdown"
        remote = getattr(parser, "parser_name", None) == "mineru"
        # 显式传入的 timeout_seconds 同时作为两段的单段预算（兼容旧语义）。
        self._locate_timeout = (
            timeout_seconds if timeout_seconds is not None else (120.0 if remote else 45.0)
        )
        self._parse_timeout = (
            parse_timeout_seconds
            if parse_timeout_seconds is not None
            else (timeout_seconds if timeout_seconds is not None else (600.0 if remote else 45.0))
        )
        # MinerU 轮询预算为 600 秒，不能被旧的 90 秒整体超时提前截断。
        self._timeout_seconds = self._locate_timeout + self._parse_timeout
        self._download_slots = asyncio.Semaphore(max(1, download_concurrency))
        self._parse_concurrency = max(1, parse_concurrency)
        self._parse_slots = asyncio.Semaphore(self._parse_concurrency)
        # Markdown 缓存命中的论文无需重新解析，转换阶段直接跳过。
        self._cache_hits: set[object] = set()

    async def download(self, paper: Paper) -> Paper:
        """阶段一：定位并下载公开全文（已有有效 Markdown 缓存时短路）。

        Raises:
            Exception: 该篇未能取得公开全文，错误原样抛出交由编排器降级。
        """
        (ready, path, error) = await self._locate_guarded(paper, self._download_slots)
        if error is not None:
            raise error
        if path is None:
            self._cache_hits.add(ready.paper_id)
        else:
            # 下载成功即登记本地路径：解析失败也保留 PDF 供重试或人工下载。
            ready.fulltext.local_pdf_path = str(path)
        return ready

    async def convert(self, paper: Paper) -> Paper:
        """阶段二：把已下载的 PDF 转换成 Markdown。"""
        if paper.paper_id in self._cache_hits or not paper.fulltext.local_pdf_path:
            return paper
        error = await self._parse_guarded(
            paper, Path(paper.fulltext.local_pdf_path), self._parse_slots
        )
        if error is not None:
            raise error
        return paper

    async def process(self, paper: Paper) -> Paper:
        """Process a single paper, re-raising its typed failure."""
        await self.download(paper)
        return await self.convert(paper)

    async def process_many(
        self,
        papers: list[Paper],
        *,
        download_concurrency: int = 4,
        parse_concurrency: int | None = None,
    ) -> list[tuple[Paper, Exception | None]]:
        """段落化处理一批论文：先并发定位/下载，再受限并发解析。

        下载是 I/O 密集、解析是远端排队，两者共用一个并发上限会让慢解析长期
        占住名额。这里分两段执行，并按入参顺序返回 ``(paper, error)``。
        """
        locate_slots = asyncio.Semaphore(max(1, download_concurrency))
        parse_slots = asyncio.Semaphore(
            max(1, parse_concurrency if parse_concurrency is not None else self._parse_concurrency)
        )
        located = await asyncio.gather(
            *(self._locate_guarded(paper, locate_slots) for paper in papers)
        )
        parse_jobs = [
            (index, paper, path)
            for index, (paper, path, error) in enumerate(located)
            if error is None and path is not None
        ]
        parse_errors = await asyncio.gather(
            *(self._parse_guarded(paper, path, parse_slots) for _, paper, path in parse_jobs)
        )
        error_by_index = {
            index: error for (index, _, _), error in zip(parse_jobs, parse_errors, strict=True)
        }
        return [
            (paper, error if error is not None else error_by_index.get(index))
            for index, (paper, _path, error) in enumerate(located)
        ]

    async def _locate_guarded(
        self, paper: Paper, slots: asyncio.Semaphore
    ) -> tuple[Paper, Path | None, Exception | None]:
        async with slots:
            try:
                async with asyncio.timeout(self._locate_timeout):
                    _parsed, path = await self._locate_or_use_local(paper)
                return paper, path, None
            except TimeoutError:
                message = (
                    f"全文定位和下载超过{self._locate_timeout:g}秒预算，保留已下载 PDF 和元数据。"
                )
                paper.fulltext.parse_error = message
                paper.fulltext.attempts.append(
                    {
                        "stage": "fulltext",
                        "status": "failed",
                        "error": message,
                        "error_type": "TimeoutError",
                        "error_kind": "provider_unavailable",
                    }
                )
                return paper, None, TimeoutError(message)
            except Exception as exc:  # noqa: BLE001 - 单篇失败交由编排器降级
                return paper, None, exc

    async def _parse_guarded(
        self, paper: Paper, pdf_path: Path, slots: asyncio.Semaphore
    ) -> Exception | None:
        async with slots:
            try:
                async with asyncio.timeout(self._parse_timeout):
                    await self._parse_pdf(paper, pdf_path)
                return None
            except TimeoutError:
                message = (
                    f"PDF 解析超过{self._parse_timeout:g}秒预算，已保留下载的 PDF，可稍后重试解析。"
                )
                paper.fulltext.parse_error = message
                paper.fulltext.attempts.append(
                    {
                        "stage": "parse",
                        "status": "failed",
                        "error": message,
                        "error_type": "TimeoutError",
                        "error_kind": "provider_unavailable",
                    }
                )
                return TimeoutError(message)
            except Exception as exc:  # noqa: BLE001 - 单篇失败交由编排器降级
                return exc

    async def _locate_or_use_local(self, paper: Paper) -> tuple[bool, Path | None]:
        """定位并下载 PDF，或直接复用已有缓存。

        Returns:
            ``(True, None)`` 表示已有有效 Markdown；``(True, path)`` 表示 PDF
            已就绪（可能是本地缓存）。

        Raises:
            NoOpenAccessCopyError: no lawful OA copy (paywall, closed,
                removed, or no URL evidence).
            ProviderUnavailableError: transport failure or invalid payload.
        """
        existing_markdown = paper.fulltext.markdown_path
        existing_pdf = paper.fulltext.local_pdf_path
        check_cache = getattr(self._parser, "cache_is_current", None)
        if existing_markdown and Path(existing_markdown).is_file():
            if check_cache is None:
                return True, None
            if (
                existing_pdf
                and Path(existing_pdf).is_file()
                and await asyncio.to_thread(
                    check_cache, Path(existing_pdf), Path(existing_markdown)
                )
            ):
                return True, None
        if existing_pdf:
            path = Path(existing_pdf).resolve()
            if path.is_file() and path.is_relative_to(self._pdf_dir.resolve()):
                return True, path
        pdf_path = None
        last_error = None
        attempted: set[str] = set()
        try:
            async with aclosing(self._candidate_batches(paper)) as batches:
                async for channel, urls in batches:
                    for candidate in urls:
                        url = str(candidate)
                        if url in attempted or not url.startswith(("https://", "http://")):
                            continue
                        attempted.add(url)
                        paper.fulltext.pdf_url = url
                        try:
                            pdf_path = await self._downloader.download(paper, self._pdf_dir)
                            paper.fulltext.attempts.append(
                                {
                                    "stage": "download",
                                    "url": url,
                                    "channel": channel,
                                    "status": "succeeded",
                                }
                            )
                            break
                        except Exception as exc:  # noqa: BLE001 - 单个副本失败，继续其他候选及渠道
                            last_error = exc
                            paper.fulltext.attempts.append(
                                {
                                    "stage": "download",
                                    "url": url,
                                    "channel": channel,
                                    "status": "failed",
                                    "error": str(exc)[:400],
                                }
                            )
                    if pdf_path is not None:
                        break
        except Exception as exc:  # noqa: BLE001 - 保留定位失败，供摘要降级使用
            last_error = last_error or exc
        if pdf_path is None:
            if attempted or paper.fulltext.access_status == AccessStatus.UNKNOWN:
                paper.fulltext.access_status = AccessStatus.UNAVAILABLE
            last_error = last_error or NoOpenAccessCopyError("No usable PDF candidate URL")
            paper.fulltext.parse_error = f"{type(last_error).__name__}: {last_error}"
            raise last_error
        return True, Path(pdf_path)

    async def _candidate_batches(self, paper: Paper) -> AsyncIterator[tuple[str, list[str]]]:
        """按渠道分批产出候选 URL，渠道名用于说明全文来自何处。

        优先使用分渠道接口，同时兼容只实现 resolve/discover 的联调组件。
        """
        staged = getattr(self._resolver, "iter_candidate_batches_labelled", None)
        if callable(staged):
            async with aclosing(staged(paper)) as batches:
                async for channel, urls in batches:
                    yield channel, urls
            return
        staged = getattr(self._resolver, "iter_candidate_batches", None)
        if callable(staged):
            async with aclosing(staged(paper)) as batches:
                async for urls in batches:
                    yield "", urls
            return
        resolution_error = None
        resolved = paper
        try:
            resolved = await self._resolver.resolve(paper)
        except Exception as exc:  # noqa: BLE001 - 有 discover 时仍允许补充定位
            resolution_error = exc
            paper.fulltext.attempts.append(
                {"stage": "resolve", "status": "failed", "error": str(exc)[:400]}
            )
        yield (
            "direct",
            [
                str(url)
                for url in [resolved.fulltext.pdf_url, *resolved.fulltext.candidate_urls]
                if url
            ],
        )
        discover = getattr(self._resolver, "discover", None)
        if callable(discover):
            await discover(paper)
            yield (
                "landing_page",
                [
                    str(url)
                    for url in [paper.fulltext.pdf_url, *paper.fulltext.candidate_urls]
                    if url
                ],
            )
        elif resolution_error is not None:
            raise resolution_error

    async def _parse_pdf(self, paper: Paper, pdf_path: Path) -> Paper:
        # 下载成功即保存原始 PDF；解析失败也要允许用户下载和重试。
        paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
        paper.fulltext.local_pdf_path = str(pdf_path)
        paper.fulltext.sha256 = await asyncio.to_thread(_sha256_file, pdf_path)
        paper.fulltext.downloaded_at = paper.fulltext.downloaded_at or datetime.now(UTC)
        paper.fulltext.markdown_path = None
        paper.fulltext.parse_quality = None
        paper.fulltext.parse_warnings = []
        parser = self._parser
        degraded = False
        try:
            markdown_path, quality = await parser.parse(pdf_path, self._markdown_dir)
        except Exception as exc:
            paper.fulltext.attempts.append(
                {
                    "stage": "parse",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                    "error_type": type(exc).__name__,
                }
            )
            if self._fallback_parser is None:
                paper.fulltext.parse_error = f"{type(exc).__name__}: {exc}"
                raise
            # 远端解析不可用时降级到本地解析：宁可少几个公式图表，也不要把
            # 已经下载成功的 PDF 记成“转 Markdown 失败”。
            parser = self._fallback_parser
            try:
                markdown_path, quality = await parser.parse(pdf_path, self._markdown_dir)
            except Exception as fallback_exc:
                paper.fulltext.parse_error = (
                    f"{type(exc).__name__}: {exc}；"
                    f"本地降级失败 {type(fallback_exc).__name__}: {fallback_exc}"
                )
                paper.fulltext.attempts.append(
                    {
                        "stage": "parse",
                        "status": "failed",
                        "error": paper.fulltext.parse_error[:400],
                        "error_type": type(fallback_exc).__name__,
                    }
                )
                raise fallback_exc from exc
            degraded = True
            paper.fulltext.attempts.append(
                {
                    "stage": "parse",
                    "status": "succeeded",
                    "reason": (
                        f"远端解析失败（{type(exc).__name__}），"
                        f"已回退本地 {getattr(parser, 'parser_name', 'pypdf')}"
                    ),
                }
            )
        else:
            paper.fulltext.attempts.append(
                {
                    "stage": "parse",
                    "status": "succeeded",
                    "reason": (
                        f"PDF 转 Markdown 完成，解析器={getattr(parser, 'parser_name', 'unknown')}"
                    ),
                }
            )
        paper.fulltext.markdown_path = str(markdown_path)
        try:
            meta = json.loads(Path(markdown_path).with_suffix(".parse.json").read_text("utf-8"))
            paper.fulltext.parse_warnings = [str(w) for w in meta.get("warnings", [])]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if degraded:
            paper.fulltext.parse_warnings.append(
                "远端解析不可用，本次 Markdown 由本地解析器生成，公式与表格可能不完整。"
            )
        parser_name = getattr(parser, "parser_name", None)
        parser_version = getattr(parser, "parser_version", None)
        if parser_name:
            paper.fulltext.parser_name = parser_name
        if parser_version:
            paper.fulltext.parser_version = parser_version
        paper.fulltext.parse_quality = quality
        paper.fulltext.parse_error = None
        return paper
