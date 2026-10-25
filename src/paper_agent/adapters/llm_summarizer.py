"""Group 3: evidence-linked single-paper summaries.

Primary path calls an OpenAI-compatible LLM with a strict JSON contract; the
reply is extracted, validated with pydantic, and retried a bounded number of
times. When no LLM client is configured, or every attempt fails validation,
the summarizer degrades to an extractive fallback that quotes sentences from
the available text — it never invents experimental details.

``generated_from_fulltext`` reflects the evidence actually used: ``True``
only when the summary was produced from parsed full text, ``False`` for
abstract-only degradation. Every produced summary carries ``EvidenceRef``
entries pointing at the section the material came from.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paper_agent.adapters.llm_client import ChatClient, extract_json_object
from paper_agent.adapters.markdown_parser import _canonical_section, split_sections
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import EvidenceRef, Paper, PaperSummary, ResearchIdea

PROMPT_VERSION = "group3-summary-v1"
MAX_QUOTE_CHARS = 500
NOT_STATED = "材料中未明确说明。"
_EVIDENCE_MISSING = "无法获取摘要或全文，未生成实质性总结。"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
#: Body characters required before a references heading is trusted.
_MIN_BODY_BEFORE_REFERENCES = 1500


def _drop_references(text: str) -> str:
    """Cut the bibliography so it cannot crowd conclusions out of the budget.

    Only a references heading preceded by a real body counts, so a stray
    early "References" line never discards the paper content. Position is
    measured in characters: reference lists are many short lines, so line
    counts would misjudge where the body ends.
    """
    offset = 0
    for line in text.splitlines(keepends=True):
        if offset >= _MIN_BODY_BEFORE_REFERENCES and _canonical_section(line) == "references":
            return text[:offset].rstrip() + "\n"
        offset += len(line)
    return text


def _truncate_middle(text: str, limit: int) -> str:
    """Keep head and tail of long text; abstracts/conclusions live there."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return text[:head] + "\n\n[...]\n\n" + text[-tail:]


def _first_sentences(text: str, *, limit: int = 2, max_chars: int = 400) -> str:
    if not text.strip():
        return ""
    flattened = re.sub(r"\s+", " ", text).strip()
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(flattened) if part.strip()]
    return " ".join(sentences[:limit])[:max_chars]


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "been",
        "into",
        "onto",
        "over",
        "under",
        "our",
        "their",
        "its",
        "using",
        "based",
        "via",
        "than",
        "then",
        "also",
        "such",
        "these",
        "those",
        "which",
    }
)


def _words(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}", text)
    } - _STOPWORDS


class _SummaryPayload(BaseModel):
    """Validated shape of the LLM's JSON reply (extra keys ignored)."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    core_problem: str = ""
    method: str = ""
    experiments: str = ""
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevance_to_idea: str = ""
    evidence: list[Any] = Field(default_factory=list)


class LlmSummarizer:
    """``PaperSummarizer`` implementation: LLM first, extractive fallback."""

    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        chat_client: ChatClient | None = None,
        max_input_chars: int = 24000,
        max_llm_attempts: int = 2,
    ) -> None:
        self._client = chat_client
        self._max_input_chars = max_input_chars
        self._max_llm_attempts = max(1, max_llm_attempts)

    async def summarize(self, idea: ResearchIdea, paper: Paper) -> PaperSummary:
        """Create an evidence-linked structured summary for one paper."""
        text, from_fulltext = await self._evidence_text(paper)
        if self._client is not None and text:
            for _ in range(self._max_llm_attempts):
                try:
                    return await self._llm_summary(idea, paper, text, from_fulltext)
                except (ProviderUnavailableError, ValidationError, ValueError):
                    continue
        return self._extractive_summary(idea, paper, text or "", from_fulltext)

    # ------------------------------------------------------------- evidence

    async def _evidence_text(self, paper: Paper) -> tuple[str | None, bool]:
        """Return ``(source_text, from_fulltext)`` choosing full text first."""
        markdown_path = paper.fulltext.markdown_path
        if markdown_path:
            path = Path(markdown_path)
            if path.exists():
                text = await asyncio.to_thread(path.read_text, "utf-8", "replace")
                if text.strip():
                    return _drop_references(text), True
        if paper.abstract and paper.abstract.strip():
            return paper.abstract.strip(), False
        return None, False

    @staticmethod
    def _to_evidence(paper: Paper, raw_items: list[Any]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            section = item.get("section")
            page = item.get("page")
            quote = item.get("quote")
            anchor = item.get("markdown_anchor")
            try:
                refs.append(
                    EvidenceRef(
                        paper_id=paper.paper_id,
                        section=str(section)[:200] if section else None,
                        page=int(page) if isinstance(page, int) and page > 0 else None,
                        quote=str(quote)[:MAX_QUOTE_CHARS] if quote else None,
                        markdown_anchor=str(anchor)[:200] if anchor else None,
                    )
                )
            except ValidationError:
                continue
        return refs

    # ------------------------------------------------------------- LLM path

    async def _llm_summary(
        self, idea: ResearchIdea, paper: Paper, text: str, from_fulltext: bool
    ) -> PaperSummary:
        source_label = "已解析全文（Markdown）" if from_fulltext else "仅有摘要"
        system = (
            "你是严谨的学术文献分析助手。只依据给定材料作答，"
            "不得虚构材料中不存在的实验设置、数据或结论；"
            "材料未覆盖的字段填写“材料中未明确说明。”；"
            "只输出一个 JSON 对象，不要输出任何多余文字。"
        )
        user = (
            f"研究想法：{idea.text}\n\n"
            f"论文题目：{paper.title}\n"
            f"证据来源：{source_label}\n\n"
            f"材料：\n{_truncate_middle(text, self._max_input_chars)}\n\n"
            "请输出如下 JSON（字段含义：core_problem 核心问题；method 方法；"
            "experiments 实验/评估；contributions 主要贡献列表；limitations 局限列表；"
            "relevance_to_idea 与该研究想法的相关性；evidence 证据列表，"
            "每项含 section（章节名）、page（页码，可选）、quote（材料原文摘录，"
            f"不超过 {MAX_QUOTE_CHARS} 字））：\n"
            '{"core_problem": "...", "method": "...", "experiments": "...", '
            '"contributions": ["..."], "limitations": ["..."], '
            '"relevance_to_idea": "...", '
            '"evidence": [{"section": "Abstract", "page": 1, "quote": "..."}]}'
        )
        raw = await self._client.complete(  # type: ignore[union-attr]
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        payload = _SummaryPayload.model_validate(extract_json_object(raw))
        if not (
            payload.core_problem.strip() or payload.method.strip() or payload.experiments.strip()
        ):
            raise ValueError("LLM summary payload is empty")
        return PaperSummary(
            paper_id=paper.paper_id,
            core_problem=payload.core_problem or NOT_STATED,
            method=payload.method or NOT_STATED,
            experiments=payload.experiments or NOT_STATED,
            contributions=[item for item in payload.contributions if item.strip()],
            limitations=[item for item in payload.limitations if item.strip()],
            relevance_to_idea=payload.relevance_to_idea or NOT_STATED,
            evidence=self._to_evidence(paper, payload.evidence),
            generated_from_fulltext=from_fulltext,
        )

    # ----------------------------------------------------- extractive path

    def _extractive_summary(
        self, idea: ResearchIdea, paper: Paper, text: str, from_fulltext: bool
    ) -> PaperSummary:
        """Offline fallback: quote sentences, never compose new claims."""
        if not text.strip():
            return PaperSummary(
                paper_id=paper.paper_id,
                core_problem=_EVIDENCE_MISSING,
                method=_EVIDENCE_MISSING,
                experiments=_EVIDENCE_MISSING,
                contributions=[],
                limitations=[],
                relevance_to_idea=_EVIDENCE_MISSING,
                evidence=[],
                generated_from_fulltext=False,
            )
        if from_fulltext:
            sections = split_sections(text)
        else:
            sections = {"abstract": text}

        def pick(*names: str) -> str:
            for name in names:
                value = sections.get(name, "")
                if value.strip():
                    return value
            return ""

        core = _first_sentences(pick("abstract", "introduction", "body")) or NOT_STATED
        method = _first_sentences(pick("method")) or NOT_STATED
        experiments = _first_sentences(pick("experiments", "analysis")) or NOT_STATED
        conclusion_text = pick("conclusion", "discussion")
        contributions = [
            sentence[:300]
            for sentence in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", conclusion_text).strip())
            if sentence.strip()
        ][:3]
        limitations = [
            sentence[:300]
            for sentence in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", pick("limitations")).strip())
            if sentence.strip()
        ][:3]
        overlap = sorted(_words(idea.text) & _words(text))[:8]
        if overlap:
            relevance = (
                "抽取式降级（未使用 LLM）：材料与研究想法的共同术语："
                + "、".join(overlap)
                + "；相关性需人工复核。"
            )
        else:
            relevance = "抽取式降级（未使用 LLM）：未识别到与想法的直接术语重叠，需人工复核。"

        evidence: list[EvidenceRef] = []
        used_sections: list[tuple[str, str]] = []
        if from_fulltext:
            for name, value in (
                ("abstract", pick("abstract", "introduction", "body")),
                ("method", pick("method")),
                ("experiments", pick("experiments", "analysis")),
                ("conclusion", conclusion_text),
                ("limitations", pick("limitations")),
            ):
                if value.strip():
                    used_sections.append((name, value))
        else:
            used_sections.append(("abstract", text))
        for name, value in used_sections:
            evidence.append(
                EvidenceRef(
                    paper_id=paper.paper_id,
                    section=name,
                    quote=re.sub(r"\s+", " ", value).strip()[:200] or None,
                )
            )
        return PaperSummary(
            paper_id=paper.paper_id,
            core_problem=core,
            method=method,
            experiments=experiments,
            contributions=contributions,
            limitations=limitations,
            relevance_to_idea=relevance,
            evidence=evidence,
            generated_from_fulltext=from_fulltext,
        )
