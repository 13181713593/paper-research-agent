"""Optional LLM-assisted concept extractor (DMFDS-inspired, not copied).

Pattern borrowed from LLM-assisted ingestion pipelines: strict JSON-schema
prompt -> parse -> on any failure (service down, bad JSON, schema mismatch)
degrade transparently to the heuristic extractor. The planner never sees an
exception from this class.

Differences from the reference project:
- No hardcoded API key; reads PAPER_AGENT_LLM_* environment variables and is
  inactive unless PAPER_AGENT_LLM_API_KEY is set.
- OpenAI-compatible ``/chat/completions`` endpoint (works with DeepSeek,
    local Ollama ``/v1``, vLLM, etc.), configurable via env.
- The LLM may only REWRITE keywords; inclusion/exclusion criteria stay
  rule-derived so user constraints are never dropped.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests

from paper_agent.adapters._concept_extraction import (
    ExtractedConcepts,
    HeuristicConceptExtractor,
)

_SYSTEM_PROMPT = (
    "You are a research literature search planner. You turn a research idea "
    "(possibly in Chinese) into English academic search keywords. Reply with "
    'strict JSON only: {"concepts": ["..."], "synonyms": ["..."]}. '
    "concepts: 3-6 English keyword phrases (multi-word preferred over single "
    "words), ordered by importance. synonyms: 4-8 alternative academic phrases, "
    "spelling variants and full forms of abbreviations used for the SAME task. "
    "Prefer specific multi-word expressions; keep application/domain qualifiers "
    "so expansion does not drift into a broader unrelated field. Do not add "
    "methods, models or datasets merely associated with the task. "
    "Never explain, never use markdown."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMConceptExtractor:
    """LLM-first extractor with transparent heuristic fallback.

    Inactive (yields to heuristic) when no API key is configured, which keeps
    the default planner assembly fully offline per the QueryPlanner contract.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 1,
        fallback: HeuristicConceptExtractor | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("PAPER_AGENT_LLM_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("PAPER_AGENT_LLM_API_KEY", "")
        self.model = model or os.getenv("PAPER_AGENT_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback = fallback or HeuristicConceptExtractor()
        self._last_mode = "heuristic"
        self.last_error = None

    @property
    def active(self) -> bool:
        return bool(self.api_key)

    @property
    def last_mode(self) -> str:
        """'llm' or 'heuristic' — which path produced the last result (for tests/logs)."""
        return self._last_mode

    def extract(self, text: str) -> ExtractedConcepts:
        self.last_error = None
        if not self.active:
            return self.fallback.extract(text)

        for attempt in range(self.max_retries + 1):
            payload = self._extract_via_llm(text)
            if payload is not None:
                concepts = payload.get("concepts") or []
                synonyms = payload.get("synonyms") or []
                if (
                    isinstance(concepts, list)
                    and concepts
                    and all(isinstance(c, str) for c in concepts)
                ):
                    clean_syn = (
                        [s for s in synonyms if isinstance(s, str)]
                        if isinstance(synonyms, list)
                        else []
                    )
                    self._last_mode = "llm"
                    self.last_error = None
                    return ExtractedConcepts(
                        concepts=[c.strip() for c in concepts if c.strip()][:8],
                        synonyms=clean_syn[:12],
                    )
            if attempt < self.max_retries:
                time.sleep(1.0)

        self._last_mode = "heuristic"
        return self.fallback.extract(text)

    def _extract_via_llm(self, text: str) -> dict[str, Any] | None:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.0,
                },
                timeout=self.timeout,
            )
            if response.status_code == 400 and "temperature" in response.text.lower():
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": text},
                        ],
                    },
                    timeout=self.timeout,
                )
            if response.status_code != 200:
                self.last_error = f"模型接口返回 HTTP {response.status_code}"
                return None
            content = response.json()["choices"][0]["message"]["content"]
            match = _JSON_RE.search(content)
            if not match:
                return None
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as exc:
            self.last_error = f"模型响应不可用：{type(exc).__name__}"
            return None
