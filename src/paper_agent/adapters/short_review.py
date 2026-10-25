"""A bounded Chinese literature review, with traceable references outside the word count."""

import json
import re

from paper_agent.adapters.llm_client import extract_json_object
from paper_agent.adapters.llm_report_writer import LlmReportWriter
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import ResearchReport


def review_length(text):
    return len(re.sub(r"\s", "", text))


class ShortReviewWriter(LlmReportWriter):
    async def write(self, idea, papers, summaries):
        body = None
        mode = "模型综合"
        failure = "未配置模型"
        if self._client and papers:
            registry = {s.paper_id: s for s in summaries}
            materials = [
                {
                    "reference": i,
                    "title": p.title,
                    "abstract": p.abstract,
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
                    "写一篇300至500字符中文综述正文，目标380字符（标点、字母、数字都计入，空白不计）。"
                    "不要逐篇罗列、标题、表格或参考文献。请控制为两段、每段约180字符。"
                    "连贯讨论研究问题、方法路线与差异、现有证据与局限；说明仅有摘要或元数据的限制。"
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
                        and 300 <= review_length(candidate) <= 500
                        and len(re.findall(r"[\u4e00-\u9fff]", candidate)) >= 120
                        and citations
                        and all(1 <= int(c) <= len(papers) for c in citations)
                    ):
                        body = candidate.strip()
                        break
                    messages.append({"role": "assistant", "content": raw})
                    failure = f"输出未通过校验（正文{review_length(candidate) if isinstance(candidate, str) else 0}字符；需300–500字符及有效编号引用）"
                    messages.append(
                        {
                            "role": "user",
                            "content": failure
                            + "。请压缩并重写为约380字符中文正文，必须使用[1]等有效编号引用。",
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
            )
        return ResearchReport(
            title=f"文献综述：{idea.text}",
            executive_summary=body,
            markdown=f"# 文献综述：{idea.text}\n\n{body}\n\n"
            f"> 正文 {review_length(body)} 字；{mode}。参考文献不计入正文。\n\n"
            "## 参考文献\n\n" + ("\n\n".join(refs) or "未找到可供综述的相关文献。"),
            paper_ids=[p.paper_id for p in papers],
            evidence=self._aggregate_evidence(summaries),
            future_directions=["【系统推断】补充全文与可比较的实验条件后再验证方法差异。"],
        )

    @staticmethod
    def _fallback(idea, papers, summaries):
        if not papers:
            return (
                "未找到符合当前研究主题与筛选条件的文献，缺乏可靠材料，无法生成有证据支持的综述。"
            )
        full = sum(s.generated_from_fulltext for s in summaries)
        registry = {s.paper_id: s for s in summaries}
        body = (
            f"围绕“{idea.text[:50]}”，本次整理了{len(papers)}篇相关文献，"
            f"其中{full}篇取得全文证据，其余材料仅能支持摘要或元数据层面的分析。"
        )
        for i, p in enumerate(papers[:3], 1):
            summary = registry.get(p.paper_id)
            detail = summary.method if summary else p.abstract
            if detail and "未明确" not in detail and "未获得" not in detail:
                body += f"文献[{i}]提供的材料记述：“{detail[:55]}”。"
            else:
                body += f"文献[{i}]的题名为“{p.title[:40]}”，当前缺少足够材料核实其具体方法。"
        notes = [
            "这些条目构成当前主题的候选证据范围，但题名相似或关键词命中并不意味着采用相同的技术路线。",
            "现有材料尚不足以统一核对各项工作的训练数据、评价指标、实验设置及统计显著性，因而不能据此宣称某种方法优于其他方法。",
            "对于未取得全文的条目，应保留作者、年份、标识符和原始来源以便复查；摘要中未披露的实验细节和研究局限不作推测。",
            "本综述采用抽取式模板降级，尚未形成经模型核验的跨论文共识；完整方法差异需结合原文证据进一步整理。",
            "【系统推断】后续宜补充可公开获取的全文，按任务、数据集与评价口径建立可比对照，并核查引文出处，再讨论结论的适用范围与研究空白。",
        ]
        for note in notes:
            if review_length(body) >= 320:
                break
            body += note
        if review_length(body) > 500:
            body = re.sub(r"\s", "", body)[:499] + "。"
        return body
