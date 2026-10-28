"""Runtime configuration.

Secrets are intentionally represented only by environment-variable names here.
Real adapters should read their own API keys at construction time and must never
put secrets in ``Paper.source_records.raw_metadata`` or MCP responses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_local_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local ``.env`` file into ``os.environ``.

    Keys already present in the real environment win (no overrides), so CI
    and container settings are never shadowed. Values may be quoted; blank
    lines and ``#`` comments are skipped. Deliberately dependency-free —
    this only supports the plain ``KEY=VALUE`` shape the project documents.
    """
    target = Path(path) if path is not None else Path.cwd() / ".env"
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str = "demo"
    max_concurrency: int = 4
    request_timeout_seconds: float = 30.0
    data_dir: str = "data"
    reranker_model: str | None = None
    embedding_model: str | None = None
    reranker_local_files_only: bool = True
    # Group 3: download / parse / summarize / report
    unpaywall_email: str = ""
    download_timeout_seconds: float = 60.0
    pdf_max_bytes: int = 30_000_000
    download_retry_limit: int = 2
    download_max_redirects: int = 5
    markdown_backend: str = "auto"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_env=os.getenv("PAPER_AGENT_ENV", "demo"),
            max_concurrency=min(32, max(1, int(os.getenv("PAPER_AGENT_MAX_CONCURRENCY", "4")))),
            request_timeout_seconds=max(
                1.0, float(os.getenv("PAPER_AGENT_REQUEST_TIMEOUT_SECONDS", "30"))
            ),
            data_dir=os.getenv("PAPER_AGENT_DATA_DIR", "data"),
            reranker_model=os.getenv("PAPER_AGENT_RERANKER_MODEL") or None,
            embedding_model=os.getenv("PAPER_AGENT_EMBEDDING_MODEL") or None,
            reranker_local_files_only=os.getenv("PAPER_AGENT_RERANKER_LOCAL_ONLY", "1") != "0",
            unpaywall_email=os.getenv("UNPAYWALL_EMAIL", ""),
            download_timeout_seconds=max(
                5.0, float(os.getenv("PAPER_AGENT_DOWNLOAD_TIMEOUT_SECONDS", "60"))
            ),
            pdf_max_bytes=max(1, int(os.getenv("PAPER_AGENT_PDF_MAX_BYTES", "30000000"))),
            download_retry_limit=max(0, int(os.getenv("PAPER_AGENT_DOWNLOAD_RETRY_LIMIT", "2"))),
            download_max_redirects=max(
                0, int(os.getenv("PAPER_AGENT_DOWNLOAD_MAX_REDIRECTS", "5"))
            ),
            markdown_backend=os.getenv("PAPER_AGENT_MARKDOWN_BACKEND", "auto"),
            llm_base_url=os.getenv("PAPER_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("PAPER_AGENT_LLM_MODEL", "gpt-4o-mini"),
            llm_timeout_seconds=max(
                10.0, float(os.getenv("PAPER_AGENT_LLM_TIMEOUT_SECONDS", "120"))
            ),
        )
