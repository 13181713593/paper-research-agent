"""Runtime configuration.

Secrets are intentionally represented only by environment-variable names here.
Real adapters should read their own API keys at construction time and must never
put secrets in ``Paper.source_records.raw_metadata`` or MCP responses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str = "demo"
    max_concurrency: int = 4
    request_timeout_seconds: float = 30.0
    data_dir: str = "data"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_env=os.getenv("PAPER_AGENT_ENV", "demo"),
            max_concurrency=max(1, int(os.getenv("PAPER_AGENT_MAX_CONCURRENCY", "4"))),
            request_timeout_seconds=max(
                1.0, float(os.getenv("PAPER_AGENT_REQUEST_TIMEOUT_SECONDS", "30"))
            ),
            data_dir=os.getenv("PAPER_AGENT_DATA_DIR", "data"),
        )

