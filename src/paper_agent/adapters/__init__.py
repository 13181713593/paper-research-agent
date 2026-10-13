"""Adapter package: demo components plus real group implementations.

Group 3 (download, parse, summarize, report) adapters are assembled by
``build_processing_components`` and wired into the pipeline through the
``production`` branch of ``paper_agent.bootstrap.build_agent``. They are
imported lazily so demo runs do not require the optional ``processing``
dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paper_agent.config import Settings
from paper_agent.ports.contracts import (
    FullTextProcessor,
    PaperSummarizer,
    ReportWriter,
)

from .demo import build_demo_dependencies

if TYPE_CHECKING:
    from paper_agent.adapters.llm_client import OpenAiCompatibleChatClient

__all__ = [
    "ProcessingComponents",
    "build_demo_dependencies",
    "build_processing_components",
]


@dataclass(slots=True)
class ProcessingComponents:
    """Group 3 adapters handed to the pipeline by the bootstrap root."""

    fulltext_processor: FullTextProcessor
    summarizer: PaperSummarizer
    report_writer: ReportWriter


def build_processing_components(settings: Settings) -> ProcessingComponents:
    """Assemble the real OA-resolve / download / parse / summarize / report set."""
    from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
    from paper_agent.adapters.llm_report_writer import LlmReportWriter
    from paper_agent.adapters.llm_summarizer import LlmSummarizer
    from paper_agent.adapters.markdown_parser import PdfMarkdownParser
    from paper_agent.adapters.open_access_resolver import UnpaywallResolver
    from paper_agent.adapters.pdf_downloader import HttpPdfDownloader

    resolver = UnpaywallResolver(
        email=settings.unpaywall_email or None,
        timeout_seconds=min(30.0, settings.request_timeout_seconds),
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
    )
    processor = PdfFullTextProcessor(
        resolver=resolver,
        downloader=downloader,
        parser=parser,
        work_dir=Path(settings.data_dir) / "fulltext",
    )
    llm_client = _build_llm_client(settings)
    return ProcessingComponents(
        fulltext_processor=processor,
        summarizer=LlmSummarizer(chat_client=llm_client),
        report_writer=LlmReportWriter(chat_client=llm_client),
    )


def _build_llm_client(settings: Settings) -> OpenAiCompatibleChatClient | None:
    """Build the shared chat client, or ``None`` for extractive-only mode.

    The API key is read from the environment here (never stored in
    ``Settings``) so it cannot leak into model objects or MCP responses.
    """
    from paper_agent.adapters.llm_client import OpenAiCompatibleChatClient

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None
    return OpenAiCompatibleChatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=api_key,
        timeout_seconds=max(settings.request_timeout_seconds, 60.0),
    )
