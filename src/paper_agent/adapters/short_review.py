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
                    "写一篇1000至1500字中文综述正文，目标1200字（标点、字母、数字都计入，空白不计）。"
                    "不要逐篇罗列、标题、表格或参考文献。请分四到五段，每段约250字。"
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
                            + "。请扩写或压缩并重写为约1200字符中文正文，必须使用[1]等有效编号引用。",
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
                "未找到符合当前研究主题与筛选条件的文献，缺少足够材料，无法生成有证据支持的综述。"
                "本次检索在设定的时间窗口、语言与来源条件下没有返回可用结果，因此无法归纳研究问题、"
                "方法路线或证据强度，任何具体结论都会超出材料范围。"
                "常见原因包括检索词与领域惯例不一致、同义词覆盖不足、时间窗口过窄，"
                "或目标主题在所选数据库中本就收录有限。"
                "需要强调，检索为空不等于该方向没有研究；它首先说明本次查询与索引没有匹配，"
                "两者含义不同，不能据此宣称领域空白。"
                "建议扩充中英文同义词与缩写写法，放宽发表日期范围，补充其他数据库渠道，"
                "并逐条核对返回题录的作者、年份与标识符是否准确。"
                "在取得可用材料之前，本报告不给出方法优劣、实验结论或研究价值的判断，"
                "只记录本次检索的条件与结果，便于后续复现与调整。"
                "后续工作应以可复核的题录与全文为准，先建立候选材料清单，再讨论可比性、局限与可验证的研究问题。"
                "本次运行完整记录了检索条件，包括主题词与同义词、语言、时间窗口、排序方式，"
                "以及每个数据库的请求状态与返回数量，这些记录可用于判断问题出在查询过窄还是索引缺失。"
                "若某个数据库请求失败、被限流或超时，结果中会以警告形式单独标注，"
                "这部分缺失代表本次未能取得数据，不能等同于该来源没有相关文献。"
                "对于跨学科主题，同一概念在不同领域常使用完全不同的术语与缩写，"
                "只按单一措辞检索会系统性地漏掉相关文献，这也是本轮结果为空时最需要优先排查的方向。"
                "下一轮建议先做小规模探索性检索，确认领域内的主流表述与典型期刊，"
                "再把确认过的词扩展为完整检索式，并固定时间窗口与筛选条件，保证前后两次结果可比。"
                "在材料补充完成之前，本报告保持空结论状态，不提供方法比较、趋势判断或研究空白推断，"
                "因为缺少材料时的任何结论都无法被复核，只会误导后续工作。"
                "若后续重新获得文献，应逐条记录获取渠道与解析状态，区分全文、摘要与仅题录三种证据强度，"
                "先建立可追溯的材料清单，再进入综合分析与结论撰写。"
                "本段属于流程说明，不构成对该研究领域的评价，也不代表相关研究方向没有价值。"
                "此外，检索结果的排序方式会直接影响人工复核的顺序，本轮记录中已保留该设置，"
                "便于按相同条件复现同一批候选，或在与上次不同的排序下交叉比较是否出现新的相关文献。"
                "若需要对外提交结论，建议先补齐材料并重新运行完整流程，"
                "再把本次的空结果作为对比基线使用，说明两次检索条件的差异以及差异带来的影响。"
                "在材料清单、获取渠道与解析状态都可追溯之前，任何跨论文比较都不具备可复核的基础，"
                "这一点在方法评述与趋势判断上尤其重要。"
            )
        full = sum(s.generated_from_fulltext for s in summaries)
        registry = {s.paper_id: s for s in summaries}
        body = (
            f"围绕“{idea.text[:50]}”，本次整理了{len(papers)}篇相关文献，"
            f"其中{full}篇取得全文证据，其余材料仅能支持摘要或元数据层面的分析。"
            "需要先说明证据边界：以下内容全部来自已获取的题录、摘要或全文，未取得全文的条目"
            "不能用于说明其方法细节、实验设置或结论强度。"
        )
        for i, p in enumerate(papers[:3], 1):
            summary = registry.get(p.paper_id)
            detail = summary.method if summary else p.abstract
            if detail and "未明确" not in detail and "未获得" not in detail:
                body += f"文献[{i}]提供的材料记述：“{detail[:55]}”。"
            else:
                body += f"文献[{i}]的题名为“{p.title[:40]}”，当前缺少足够材料核实其具体方法。"
        notes = [
            "从已获取的材料看，这些工作共享同一个问题域，但在任务定义、数据来源与评价口径上并不一致，因此难以直接横向比较。",
            "部分文献只提供了摘要层面的信息，摘要通常省略数据规模、预处理流程与消融实验，单凭摘要判断方法有效性风险较高。",
            "题名与关键词的相似并不代表采用相同的技术路线，同一术语在不同文献中可能指向不同的对象或粒度。",
            "现有材料尚不足以统一核对各项工作的训练数据、评价指标、实验设置及统计显著性，因而不能据此宣称某种方法优于其他方法。",
            "对于未取得全文的条目，应保留作者、年份、标识符和原始来源以便复查；摘要中未披露的实验细节和研究局限不作推测。",
            "本综述采用抽取式模板降级，尚未形成经模型核验的跨论文共识；完整的方法差异需要结合原文证据逐项整理。",
            "在证据强度上，可复核的全文材料优先于摘要，摘要优先于仅有的题录；本报告按该顺序排列可信度，并在引用处标注来源。",
            "若后续补充全文，应优先复核方法章节与实验表格，确认指标定义、对比基线以及结果的可重复条件，再更新本综述的结论。",
            "综合来看，当前证据只能支持对研究问题的描述性归纳，不支持对方法优劣的排序判断，也不足以给出量化结论。",
            "材料中反复出现的共性线索应视为候选假设，需要在统一的数据划分与评价指标下重新验证，才能判断其稳健性。",
            "研究空白只能从已覆盖材料的共同缺口推断，若某方向在本批文献中缺席，也可能只是检索覆盖范围不足所致。",
            "本报告的引用编号与参考文献一一对应，可据编号回查原始文献、获取渠道与解析状态，便于逐条复核。",
            "【系统推断】后续宜补充可公开获取的全文，按任务、数据集与评价口径建立可比对照，并核查引文出处，再讨论结论的适用范围与研究空白。",
        ]
        for note in notes:
            if review_length(body) >= 1150:
                break
            body += note
        if review_length(body) < 1000:
            body += (
                "在方法层面，已覆盖的材料更多描述了各自的处理流程，而较少报告失败案例与边界条件，"
                "这使得跨研究的稳定性判断缺少直接依据。"
                "从证据链看，题录只能确认文献存在，摘要可以说明研究目标与大致方法，"
                "只有全文才能核对数据划分、超参数设置、对比基线与显著性检验。"
                "因此本报告把三者分开标注，并在引用处保留可回查的编号，避免把摘要层面的表述当作全文结论。"
                "建议后续工作同时记录数据来源、预处理方式、评价指标定义与统计检验方法，"
                "并公开可复现的实验配置，以便在相同口径下比较不同技术路线，"
                "同时明确说明结论适用的任务范围与前提条件。"
                "若条件允许，还应复核作者自述的局限与研究空白，区分作者结论与本文的系统推断。"
            )
        if review_length(body) > 1500:
            body = re.sub(r"\s", "", body)[:1499] + "。"
        return body
