"""共享适配器导出；第三组解析组件按需导入，保留检索组公共接口。"""

from __future__ import annotations

import os
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paper_agent.config import Settings
from paper_agent.ports.contracts import FullTextProcessor, PaperSummarizer, ReportWriter

from .arxiv_retriever import (
    ArxivError,
    ArxivNoResultsError,
    ArxivRateLimitError,
    ArxivRetriever,
    ArxivSearchConfig,
    ArxivTimeoutError,
)
from .bibtex_mapper import BibTexMapper, BibTexMappingError
from .demo import build_demo_dependencies
from .paper_merger import PaperMerger

if TYPE_CHECKING:
    from paper_agent.adapters.llm_client import OpenAiCompatibleChatClient

__all__ = [
    "ArxivError",
    "ArxivNoResultsError",
    "ArxivRateLimitError",
    "ArxivRetriever",
    "ArxivSearchConfig",
    "ArxivTimeoutError",
    "BibTexMapper",
    "BibTexMappingError",
    "PaperMerger",
    "ProcessingComponents",
    "build_demo_dependencies",
    "build_processing_components",
]


@dataclass(slots=True)
class ProcessingComponents:
    """供主流程和独立工作台复用的第三组组件。"""

    fulltext_processor: FullTextProcessor
    summarizer: PaperSummarizer
    report_writer: ReportWriter


def build_processing_components(
    settings: Settings, *, parser_executor: Executor | None = None
) -> ProcessingComponents:
    """装配开放获取定位、下载、解析、总结与报告，不发起外部请求。"""
    from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
    from paper_agent.adapters.llm_summarizer import LlmSummarizer
    from paper_agent.adapters.markdown_parser import PdfMarkdownParser
    from paper_agent.adapters.open_access_resolver import UnpaywallResolver
    from paper_agent.adapters.pdf_downloader import HttpPdfDownloader
    from paper_agent.adapters.short_review import ShortReviewWriter

    resolver = UnpaywallResolver(
        email=settings.unpaywall_email,
        timeout_seconds=min(30.0, settings.request_timeout_seconds),
        discover_locations=True,
    )
    downloader = HttpPdfDownloader(
        timeout_seconds=settings.download_timeout_seconds,
        max_bytes=settings.pdf_max_bytes,
        retry_limit=settings.download_retry_limit,
        max_redirects=settings.download_max_redirects,
    )
    parser = PdfMarkdownParser(
        backend=settings.markdown_backend,
        mineru_api_key=os.getenv("MINERU_API_KEY") or None,
        executor=parser_executor,
    )
    processor = PdfFullTextProcessor(
        resolver=resolver,
        downloader=downloader,
        parser=parser,
        work_dir=Path(settings.data_dir) / "fulltext",
    )
    llm_client = _build_llm_client(settings, allow_planner_key=True)
    return ProcessingComponents(
        fulltext_processor=processor,
        summarizer=LlmSummarizer(chat_client=llm_client),
        report_writer=ShortReviewWriter(chat_client=llm_client),
    )


def _build_llm_client(
    settings: Settings, *, allow_planner_key=False
) -> OpenAiCompatibleChatClient | None:
    """密钥仅从环境读取，不放入 Settings、模型对象或 MCP 响应。"""
    from paper_agent.adapters.llm_client import OpenAiCompatibleChatClient

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key and allow_planner_key:
        api_key = (os.getenv("PAPER_AGENT_LLM_API_KEY") or "").strip()
    if not api_key:
        return None
    return OpenAiCompatibleChatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=api_key,
        timeout_seconds=settings.llm_timeout_seconds,
    )
