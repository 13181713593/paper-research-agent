"""Group 3: synthesize the final Markdown research report.

The report is always built from validated ``PaperSummary`` objects. When an
LLM client is configured it composes the narrative; otherwise (or when every
LLM attempt fails) a deterministic template assembles the same sections.

Contract rules enforced here (see ``docs/API_CONTRACTS.md``):

- synthesis uses only ``PaperSummary`` content and their evidence;
- future directions are explicitly labeled as system inference, never as
  paper claims;
- the report aggregates the ``EvidenceRef`` entries of its summaries so
  factual claims stay traceable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paper_agent.adapters.llm_client import ChatClient, extract_json_object
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import (
    EvidenceRef,
    Paper,
    PaperSummary,
    ResearchIdea,
    ResearchReport,
)

PROMPT_VERSION = "group3-report-v1"
INFERENCE_LABEL = "【系统推断】"


class _ReportPayload(BaseModel):
    """Validated shape of the LLM's JSON reply (extra keys ignored)."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = ""
    executive_summary: str = ""
    markdown: str = ""
    future_directions: list[str] = Field(default_factory=list)


def _label_inference(direction: str) -> str:
    text = direction.strip()
    if not text:
        return text
    if text.startswith(INFERENCE_LABEL):
        return text
    return f"{INFERENCE_LABEL}{text}"


def _dump_summaries_for_prompt(
    papers: Sequence[Paper], summaries: Sequence[PaperSummary]
) -> str:
    """Compact JSON view of the summaries used in the report prompt."""
    registry = {summary.paper_id: summary for summary in summaries}
    entries = []
    for paper in papers:
        summary = registry.get(paper.paper_id)
        entries.append(
            {
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue.name if paper.venue and paper.venue.name else None,
                "summary": summary.model_dump(mode="json") if summary else None,
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)


def comparison_markdown(papers: Sequence[Paper], summaries: Sequence[PaperSummary]) -> str:
    """从结构化总结生成横向对照表，不额外推断共识或实验结论。"""
    registry = {summary.paper_id: summary for summary in summaries}

    def cell(value: str) -> str:
        text = " ".join(value.split())
        text = text[:180] + "…" if len(text) > 180 else text
        return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")

    lines = [
        "## 跨论文对比矩阵", "",
        "> 以下仅归纳已有总结；表格长字段节选展示，完整内容见单篇总结。", "",
        "| 论文 | 材料来源 | 方法 | 实验/评估 | 局限 | 引文数（未核验） |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for paper in papers:
        summary = registry.get(paper.paper_id)
        if summary is None:
            lines.append(f"| {cell(paper.title)} | 未生成总结 | — | — | — | 0 |")
            continue
        source = "全文" if summary.generated_from_fulltext else "仅摘要（降级）"
        lines.append(
            f"| {cell(paper.title)} | {source} | {cell(summary.method)} | "
            f"{cell(summary.experiments)} | {cell('；'.join(summary.limitations) or '未明确说明')} | "
            f"{len(summary.evidence)} |"
        )
    return "\n".join(lines) + "\n"


class LlmReportWriter:
    """``ReportWriter`` implementation: LLM narrative with template fallback."""

    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        chat_client: ChatClient | None = None,
        max_llm_attempts: int = 2,
    ) -> None:
        self._client = chat_client
        self._max_llm_attempts = max(1, max_llm_attempts)

    async def write(
        self,
        idea: ResearchIdea,
        papers: Sequence[Paper],
        summaries: Sequence[PaperSummary],
    ) -> ResearchReport:
        """Synthesize the Markdown report for the final Top-k papers."""
        registry = {summary.paper_id: summary for summary in summaries}
        if self._client is not None and summaries:
            for _ in range(self._max_llm_attempts):
                try:
                    return await self._llm_report(idea, papers, summaries)
                except (ProviderUnavailableError, ValidationError, ValueError):
                    continue
        return self._template_report(idea, papers, registry)

    # ------------------------------------------------------------- LLM path

    async def _llm_report(
        self,
        idea: ResearchIdea,
        papers: Sequence[Paper],
        summaries: Sequence[PaperSummary],
    ) -> ResearchReport:
        system = (
            "你是严谨的学术综述写作助手。只能基于给定的论文总结与证据引用进行综合，"
            "不得虚构论文中不存在的内容；未来方向属于你的系统推断，必须标注"
            f"“{INFERENCE_LABEL}”，不得冒充论文作者结论；只输出一个 JSON 对象。"
        )
        user = (
            f"研究想法：{idea.text}\n\n"
            f"论文总结（JSON）：\n{_dump_summaries_for_prompt(papers, summaries)}\n\n"
            "请输出如下 JSON：{\"title\": 报告标题, \"executive_summary\": 一段总体概述, "
            "\"markdown\": 完整 Markdown 报告（含研究想法、逐篇要点、共识与分歧、"
            "未来方向（系统推断，非论文结论）等章节）, "
            "\"future_directions\": [未来方向条目（每条都会被标注为系统推断）]}"
        )
        raw = await self._client.complete(  # type: ignore[union-attr]
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        payload = _ReportPayload.model_validate(extract_json_object(raw))
        if not payload.markdown.strip():
            raise ValueError("LLM report payload has empty markdown")
        title = payload.title.strip() or self._default_title(idea)
        directions = [
            _label_inference(item)
            for item in payload.future_directions
            if item and item.strip()
        ]
        return ResearchReport(
            title=title,
            executive_summary=payload.executive_summary.strip()
            or self._default_executive_summary(papers, summaries),
            markdown=(
                "> 说明：报告中的未来方向均为【系统推断】，不代表论文作者结论。\n\n"
                + payload.markdown.strip() + "\n\n"
                + comparison_markdown(papers, summaries)
            ),
            paper_ids=[paper.paper_id for paper in papers],
            future_directions=directions,
            evidence=self._aggregate_evidence(summaries),
        )

    # -------------------------------------------------------- template path

    def _template_report(
        self,
        idea: ResearchIdea,
        papers: Sequence[Paper],
        registry: dict,
    ) -> ResearchReport:
        summaries = list(registry.values())
        fulltext_count = sum(1 for item in summaries if item.generated_from_fulltext)
        lines = [
            f"# {self._default_title(idea)}",
            "",
            ("> 生成模式：模板合成（未配置 LLM 或 LLM 不可用）；"
            "所有事实来自论文总结与其证据引用。"),
            "",
            "## 研究想法",
            "",
            idea.text,
            "",
            "## 论文概览",
            "",
            (f"共 {len(papers)} 篇论文；{fulltext_count} 篇总结基于全文，"
            f"{len(summaries) - fulltext_count} 篇基于摘要（降级）。"),
            "",
        ]
        lines.extend([comparison_markdown(papers, summaries), ""])
        for index, paper in enumerate(papers, 1):
            summary = registry.get(paper.paper_id)
            lines.extend([f"### {index}. {paper.title}", ""])
            meta_bits: list[str] = []
            if paper.year:
                meta_bits.append(str(paper.year))
            if paper.venue and paper.venue.name:
                meta_bits.append(paper.venue.name)
            sources = ", ".join(sorted({record.source.value for record in paper.source_records}))
            if sources:
                meta_bits.append(f"来源：{sources}")
            if meta_bits:
                lines.append("- " + " · ".join(meta_bits))
            if paper.scores.final is not None:
                lines.append(f"- 相关性评分：{paper.scores.final:.3f}")
            if summary is None:
                lines.extend(["- 未生成总结。", ""])
                continue
            status = "全文证据" if summary.generated_from_fulltext else "仅摘要（降级）"
            lines.extend([f"- 证据状态：{status}", ""])
            lines.extend([f"**核心问题：** {summary.core_problem}", ""])
            lines.extend([f"**方法：** {summary.method}", ""])
            lines.extend([f"**实验/评估：** {summary.experiments}", ""])
            if summary.contributions:
                lines.extend(["**主要贡献：**"])
                lines.extend(f"- {item}" for item in summary.contributions)
                lines.append("")
            if summary.limitations:
                lines.extend(["**局限：**"])
                lines.extend(f"- {item}" for item in summary.limitations)
                lines.append("")
            lines.extend([f"**与想法相关性：** {summary.relevance_to_idea}", ""])
        directions = self._template_directions(summaries)
        lines.extend(["## 未来方向（系统推断，非论文结论）", ""])
        lines.extend(f"- {item}" for item in directions)
        lines.append("")
        return ResearchReport(
            title=self._default_title(idea),
            executive_summary=self._default_executive_summary(papers, summaries),
            markdown="\n".join(lines),
            paper_ids=[paper.paper_id for paper in papers],
            future_directions=directions,
            evidence=self._aggregate_evidence(summaries),
        )

    @staticmethod
    def _template_directions(summaries: list[PaperSummary]) -> list[str]:
        seen: list[str] = []
        for summary in summaries:
            for limitation in summary.limitations[:1]:
                normalized = limitation.strip()
                if normalized and normalized not in seen:
                    seen.append(normalized)
        if not summaries:
            return [_label_inference("未获得可用总结，无法给出方向建议。")]
        directions = [f"围绕已识别局限继续探索：{item}" for item in seen[:4]]
        if not directions:
            return [
                _label_inference(
                    "当前总结未提供足以形成方向建议的局限信息；建议扩充候选文献后重新评估。"
                )
            ]
        return [_label_inference(item) for item in directions]

    @staticmethod
    def _default_title(idea: ResearchIdea) -> str:
        short = " ".join(idea.text.split())
        return f"研究报告：{short[:60]}"

    @staticmethod
    def _default_executive_summary(
        papers: Sequence[Paper], summaries: Sequence[PaperSummary]
    ) -> str:
        fulltext_count = sum(1 for item in summaries if item.generated_from_fulltext)
        return (
            f"本报告基于 {len(papers)} 篇 Top-k 论文的 {len(summaries)} 份结构化总结"
            f"（其中 {fulltext_count} 份基于全文证据，"
            f"{len(summaries) - fulltext_count} 份为摘要降级）综合生成。"
        )

    @staticmethod
    def _aggregate_evidence(summaries: Sequence[PaperSummary]) -> list[EvidenceRef]:
        return [ref for summary in summaries for ref in summary.evidence]
