"""A bounded Chinese literature review, with traceable references outside the word count."""

import json
import re

from paper_agent.adapters.llm_client import extract_json_object
from paper_agent.adapters.llm_report_writer import (
    LlmReportWriter,
    comparison_markdown,
    publication_note,
)
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import ResearchReport


def review_length(text):
    return len(re.sub(r"\s", "", text))


class ShortReviewWriter(LlmReportWriter):
    async def write(self, idea, papers, summaries):
        body = None
        mode = "模型综合"
        failure = "未配置模型"
        quality_notes = []
        if self._client and papers:
            registry = {s.paper_id: s for s in summaries}
            materials = [
                {
                    "reference": i,
                    "title": p.title,
                    "abstract": p.abstract,
                    "publication_note": publication_note(p),
                    "summary": registry[p.paper_id].model_dump(mode="json")
                    if p.paper_id in registry
                    else None,
                }
                for i, p in enumerate(papers, 1)
            ]
            messages = [
                {
                    "role": "system",
                    "content": "你是严谨的中文学术综述作者。只根据提供的文献材料写作，"
                    "不编造方法、实验数字、优劣结论或共识。用[1]等编号引用。仅输出JSON。",
                },
                {
                    "role": "user",
                    "content": f"研究主题：{idea.text}\n"
                    f"材料：{json.dumps(materials, ensure_ascii=False)}\n"
                    "写一篇1000至1500字中文综述正文，目标1200字（标点、字母、数字都计入，空白不计）。"
                    "不要逐篇罗列、标题、表格或参考文献。请分四段，每段约250字符。"
                    "正文不要重复英文论文标题，长术语用中文名称表达，算法保留简短缩写；完整题名另列参考文献。"
                    "连贯讨论研究问题、方法路线与差异、现有证据与局限；说明仅有摘要或元数据的限制。"
                    "必须回应用户的具体比较重点。若主题涉及云/边/雾与资源调度，应按架构层级、"
                    "决策变量、资源约束、优化目标、算法与适用场景进行实质对比，不能只泛谈证据局限。"
                    "禁止将不同实验条件下的百分比横向拼成方法排名。若有预印本，说明本次未核验同行评审。"
                    "正文须覆盖并引用全部已选论文；只有题录的文献说明证据不足，不据题目推断算法。"
                    "summary.generated_from_fulltext=true代表总结来自已解析全文，不得称该文献仅有摘要或元数据；"
                    "总结未记述某细节，只能说明本次材料未涵盖，不能断言论文没有该细节。"
                    '未来建议标注【系统推断】，不冒充作者结论。输出{"review":"综述正文"}。',
                },
            ]
            for _ in range(self._max_llm_attempts):
                try:
                    raw = await self._client.complete(messages, temperature=0.2)
                    candidate = extract_json_object(raw).get("review", "")
                    citations = (
                        re.findall(r"\[(\d+)\]", candidate) if isinstance(candidate, str) else []
                    )
                    if (
                        isinstance(candidate, str)
                        and 1000 <= review_length(candidate) <= 1500
                        and len(re.findall(r"[\u4e00-\u9fff]", candidate)) >= 700
                        and citations
                        and all(1 <= int(c) <= len(papers) for c in citations)
                    ):
                        body = candidate.strip()
                        break
                    messages.append({"role": "assistant", "content": raw})
                    failure = f"输出未通过校验（正文{review_length(candidate) if isinstance(candidate, str) else 0}字符；需1000–1500字符及有效编号引用）"
                    messages.append(
                        {
                            "role": "user",
                            "content": failure
                            + "。请重写为约1050字符中文正文。英文也逐字符计数，请删除重复背景与长英文名称，"
                            "保留具体方法差异、架构约束与[1]等有效编号引用。",
                        }
                    )
                except ProviderUnavailableError:
                    failure = "模型接口不可用或超时"
                    continue
                except (ValueError, TypeError):
                    failure = "模型未返回有效JSON正文"
                    continue
        if body is None:
            body = self._fallback(idea, papers, summaries)
            mode = f"抽取式模板合成（{failure}）"
            quality_notes.append(
                "未生成模型综述：以下是原文材料摘录与证据对照，不是已完成的跨论文综合分析。"
            )
        registry = {s.paper_id: s for s in summaries}
        full_count = sum(
            bool(registry.get(p.paper_id) and registry[p.paper_id].generated_from_fulltext)
            for p in papers
        )
        if full_count < len(papers):
            quality_notes.append(
                f"全文证据 {full_count}/{len(papers)} 篇；缺失全文的实验细节与结论强度待核验。"
            )
        if any(s.generation_method in {"extractive", "no_evidence"} for s in summaries):
            quality_notes.append(
                "部分单篇总结使用原文抽取或缺少材料；不能将未提取到的信息解释为论文没有该方法。"
            )
        refs = []
        for i, paper in enumerate(papers, 1):
            url = (
                f"https://doi.org/{paper.doi}"
                if paper.doi
                else (
                    f"https://arxiv.org/abs/{paper.arxiv_id}"
                    if paper.arxiv_id
                    else paper.landing_page_url
                )
            )
            refs.append(
                f"[{i}] {paper.title}（{paper.year or '年份未知'}）"
                + (f" [来源]({url})" if url else "")
                + (f" · {publication_note(paper)}" if publication_note(paper) else "")
            )
        return ResearchReport(
            title=f"文献综述：{idea.text}",
            executive_summary=body,
            markdown=f"# 文献综述：{idea.text}\n\n{body}\n\n"
            f"> 正文 {review_length(body)} 字；{mode}。参考文献不计入正文。\n\n"
            + ("\n".join(f"> {note}" for note in quality_notes) + "\n\n" if quality_notes else "")
            + comparison_markdown(papers, summaries)
            + "\n## 参考文献\n\n"
            + ("\n\n".join(refs) or "未找到可供综述的相关文献。"),
            paper_ids=[p.paper_id for p in papers],
            evidence=self._aggregate_evidence(summaries),
            future_directions=["【系统推断】补充全文与可比较的实验条件后再验证方法差异。"],
            generation_method="model" if mode == "模型综合" else "extractive",
            quality_notes=quality_notes,
        )

    @staticmethod
    def _fallback(idea, papers, summaries):
        """Use available material; never pad an unavailable analysis to a word quota."""
        if not papers:
            return "未找到可用文献，缺少足够材料，无法生成综述。检索为空不代表领域没有研究，请核对来源状态与检索条件。"
        registry = {s.paper_id: s for s in summaries}
        full = sum(
            bool(registry.get(p.paper_id) and registry[p.paper_id].generated_from_fulltext)
            for p in papers
        )
        lines = [
            (
                f"本次整理了{len(papers)}篇文献，其中{full}篇有全文证据。以下是原文材料摘录，不是模型综合结论；"
                "缺少足够材料的字段保留空缺，完整逐篇对照见下表。"
            )
        ]
        for index, paper in enumerate(papers, 1):
            summary = registry.get(paper.paper_id)
            if summary and summary.generation_method != "no_evidence":
                detail = summary.method
                if not detail or "未明确" in detail or "无法获取" in detail:
                    detail = paper.abstract or "本次未提取到方法证据，需回查原文。"
            else:
                detail = paper.abstract or "仅有题录，缺少摘要与全文，无法分析方法。"
            excerpt = " ".join(detail.split())
            excerpt = excerpt[:160] + "…（节选）" if len(excerpt) > 160 else excerpt
            line = f"文献[{index}]：{excerpt}"
            if review_length("\n\n".join([*lines, line])) > 1300:
                break
            lines.append(line)
        lines.append(
            "当前材料不能支持跨论文的量化优劣排名。不同论文的架构、工作负载、约束和指标口径须逐项核对，"
            "预印本的同行评审状态也需另行确认。材料缺失不等于论文没有相应方法或实验。"
            "【系统推断】补齐证据并配置可用模型后，才能进一步形成针对研究问题的综合分析。"
        )
        return "\n\n".join(lines)
