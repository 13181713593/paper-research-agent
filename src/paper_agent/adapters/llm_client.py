"""Shared async client for OpenAI-compatible chat completion APIs.

Group 3 reuses ``httpx`` (already a transitive dependency through ``mcp``)
instead of vendor SDKs. API keys are read by the owning adapters at
construction time and never stored in ``Settings``, logs, or model objects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from paper_agent.domain.errors import ProviderUnavailableError


class ChatClient(Protocol):
    """Minimal chat interface so summarizer/report tests can inject fakes."""

    async def complete(
        self, messages: list[dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        """Return the assistant text for one chat completion round-trip."""
        ...


class OpenAiCompatibleChatClient:
    """``POST {base_url}/chat/completions`` against any OpenAI-compatible API.

    Works with OpenAI, DashScope, DeepSeek, vLLM, Ollama, ... by pointing
    ``PAPER_AGENT_LLM_BASE_URL`` at the compatible endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._transport = transport

    async def complete(
        self, messages: list[dict[str, str]], *, temperature: float = 0.2
    ) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"LLM request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"LLM endpoint returned HTTP {response.status_code}"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(f"Malformed LLM response: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderUnavailableError("LLM returned empty content")
        return content


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from a model reply.

    Handles fenced code blocks and leading/trailing prose. Raises ``ValueError``
    when no JSON object can be recovered so callers can retry or degrade.
    """
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in model output: {exc}") from exc
    if not isinstance(parsed, dict):
        # ValueError on purpose: callers treat every malformed reply uniformly
        # (``except ValueError``) and degrade to their offline fallback.
        raise ValueError("model output is not a JSON object")  # noqa: TRY004
    return parsed
