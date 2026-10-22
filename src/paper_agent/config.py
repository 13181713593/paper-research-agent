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
    reranker_local_files_only: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_env=os.getenv("PAPER_AGENT_ENV", "demo"),
            max_concurrency=max(1, int(os.getenv("PAPER_AGENT_MAX_CONCURRENCY", "4"))),
            request_timeout_seconds=max(
                1.0, float(os.getenv("PAPER_AGENT_REQUEST_TIMEOUT_SECONDS", "30"))
            ),
            data_dir=os.getenv("PAPER_AGENT_DATA_DIR", "data"),
            reranker_model=os.getenv("PAPER_AGENT_RERANKER_MODEL") or None,
            reranker_local_files_only=os.getenv("PAPER_AGENT_RERANKER_LOCAL_ONLY", "1") != "0",
        )

