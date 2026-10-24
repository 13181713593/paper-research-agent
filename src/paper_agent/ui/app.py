"""Streamlit UI for the shared main-agent facade."""

from __future__ import annotations

import asyncio
import os

import streamlit as st

from paper_agent.application.processing_benchmark import benchmark_to_markdown
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import PipelineRequest, ProcessingStatus, ResearchIdea
from paper_agent.ui.markdown_exports import (
    evidence_audit,
    load_paper_markdown,
    markdown_filename,
    summary_to_markdown,
    zip_markdown_files,
)
from paper_agent.ui.workbench import render_benchmark, render_processing_metrics, render_workbench

st.set_page_config(page_title="论文研究智能体", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent():
    return build_agent()


def run_async(coro):
    """Keep asynchronous orchestration behind one UI boundary."""
    return asyncio.run(coro)


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in value.replace("，", ",").split(",") if term.strip()]


st.title("多源论文检索与总结智能体")
st.caption(
    "查询规划 → 多源检索 → 合并去重 → 元数据预排序 → 全文处理 → "
    "证据确认 → 最终重排 → 单篇总结 → 综合报告"
)

load_local_env()
settings = Settings.from_env()
mode = st.radio("运行入口", ["完整研究流程", "第三组独立演示"], horizontal=True)

with st.sidebar:
    st.header("任务配置")
    top_k = st.slider("最终论文数 Top-k", 1, 20, 5, help="全文确认后最终返回的论文数")
    pool_size = st.slider(
        "全文候选池 N", top_k, 100, max(20, top_k),
        help="预排序后进入下载解析的候选数；N 应明显大于 k，以便失败补位。",
    )
    source_limit = st.slider("每个来源最多召回", 5, 200, 30)
    st.number_input("并发提示", 1, 32, 4, disabled=True)
    require_fulltext = st.checkbox("仅保留成功解析全文", value=False)
    allow_abstract_fallback = st.checkbox("全文失败时允许摘要降级", value=True)
    st.divider()
    contract = get_agent().contract()
    st.caption(f"服务版本 {contract.version} · 至少 {contract.minimum_sources} 个数据源")
    independent = mode == "第三组独立演示"
    if independent:
        st.info("独立入口：按 arXiv 编号获取元数据并真实下载、解析和总结；不执行检索或排序。")
    else:
        mode_descriptions = {
            "demo": "demo：全部使用离线演示组件，不执行真实检索或全文下载。",
            "retrieval": "retrieval：五源真实检索与去重；排序、全文处理、总结和报告仍为演示组件。",
            "ranking": "ranking：第二组真实预排序、证据确认和重排；检索、全文处理和总结仍为演示组件。",
            "production": "production：五源真实检索 → 去重 → 第二组排序 → 第三组全文处理与总结报告。",
        }
        st.info(mode_descriptions.get(settings.app_env, f"未知模式：{settings.app_env}"))
        if settings.app_env in {"retrieval", "production"}:
            if (os.getenv("PAPER_AGENT_LLM_API_KEY") or "").strip():
                st.caption(f"查询规划已配置模型 {settings.llm_model}；调用失败时使用启发式规划。")
            else:
                st.caption("未配置查询规划 API：使用启发式规划，中文查询建议补充英文关键词。")
        if settings.app_env in {"ranking", "production"} and not settings.reranker_model:
            st.caption("未配置重排模型：使用第二组规则排序，实际依据见排序分项。")
    if independent or settings.app_env == "production":
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            st.info("未配置总结 API：使用抽取式单篇总结和模板综合报告，仍可下载 Markdown。")
        else:
            st.caption(
                f"已配置总结模型：{settings.llm_model}。"
                "生成时会调用 API；调用失败时会降级为抽取式总结与模板报告。"
            )

idea = st.text_area(
    "研究想法",
    value="面向无人机视觉语言导航的长期记忆、失败恢复与世界模型联合方法",
    height=130,
    placeholder="请描述研究对象、核心问题、方法侧重点和期望时间范围……",
)

with st.expander("可选检索约束"):
    c1, c2, c3 = st.columns(3)
    required_text = c1.text_input("必须包含词（逗号分隔）")
    excluded_text = c2.text_input("排除词（逗号分隔）")
    focus_text = c3.text_input("排序侧重点（逗号分隔）", placeholder="方法相似度, 最新工作")

if mode == "第三组独立演示":
    render_workbench(settings, idea)
else:
    st.caption("点击下方按钮生成结果；完成后可下载论文全文 Markdown、单篇总结和综合报告。")
if mode == "完整研究流程" and st.button(
    "开始研究（下载论文并生成总结）", type="primary",
    disabled=len(idea.strip()) < 10, use_container_width=True,
):
    request = PipelineRequest(
        idea=ResearchIdea(
            text=idea,
            required_terms=split_terms(required_text),
            excluded_terms=split_terms(excluded_text),
            ranking_focus=split_terms(focus_text),
        ),
        top_k=top_k,
        candidate_limit_per_source=source_limit,
        pre_rank_pool_size=pool_size,
        require_fulltext=require_fulltext,
        allow_abstract_fallback=allow_abstract_fallback,
    )
    with st.status("主 Agent 正在执行……", expanded=True) as status_box:
        result = run_async(get_agent().research(request))
        for event in result.events:
            icon = "✅" if event.status.value == "succeeded" else "⚠️"
            st.write(
                f"{icon} {event.stage.value} · {event.duration_ms or 0} ms · "
                f"{event.output_count if event.output_count is not None else '—'} 条"
            )
        status_box.update(
            label=f"流程结束：{result.status.value}",
            state="complete" if result.status != ProcessingStatus.FAILED else "error",
        )
    st.session_state["pipeline_result"] = result
    st.session_state.pop("processing_run", None)
    st.session_state.pop("processing_benchmark", None)

result = st.session_state.get("pipeline_result")
if result:
    render_processing_metrics(result)
    processing_run = st.session_state.get("processing_run")
    is_workbench = bool(processing_run and processing_run.result.run_id == result.run_id)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最终论文", len(result.papers))
    m2.metric("完成阶段", len(result.events))
    m3.metric("累计阶段耗时", f"{result.total_duration_ms / 1000:.2f}s")
    m4.metric("警告", len(result.warnings))

    paper_by_id = {paper.paper_id: paper for paper in result.papers}
    paper_documents = {
        paper.paper_id: load_paper_markdown(paper, settings.data_dir)
        for paper in result.papers
    }
    summary_sources = {
        summary.paper_id: (
            paper_documents.get(summary.paper_id) if summary.generated_from_fulltext
            else (paper_by_id[summary.paper_id].abstract if summary.paper_id in paper_by_id else None)
        ) for summary in result.summaries
    }
    summary_documents = {
        summary.paper_id: summary_to_markdown(
            summary, paper_by_id.get(summary.paper_id), source_text=summary_sources[summary.paper_id],
        ) for summary in result.summaries
    }
    downloads = {}
    for paper in result.papers:
        markdown = paper_documents[paper.paper_id]
        if markdown is not None:
            filename = markdown_filename(paper.title, paper.paper_id, "paper")
            downloads[f"papers/{filename}"] = markdown
    for summary in result.summaries:
        paper = paper_by_id.get(summary.paper_id)
        title = paper.title if paper else "paper"
        filename = markdown_filename(title, summary.paper_id, "summary")
        downloads[f"summaries/{filename}"] = summary_documents[summary.paper_id]
    if result.report and result.report.markdown.strip():
        downloads["research_report.md"] = result.report.markdown
    benchmark = st.session_state.get("processing_benchmark")
    if benchmark and benchmark["run_id"] == str(result.run_id):
        downloads["benchmark.md"] = benchmark_to_markdown(benchmark["report"])

    st.subheader("Markdown 下载")
    st.download_button(
        "一键下载全部 Markdown（ZIP）",
        zip_markdown_files(downloads),
        file_name=f"research_{result.run_id}.zip", mime="application/zip",
        key=f"bundle_{result.run_id}", disabled=not downloads,
    )
    available_count = sum(text is not None for text in paper_documents.values())
    st.caption(
        f"可下载全文 {available_count}/{len(result.papers)} 篇，"
        f"单篇总结 {len(result.summaries)} 份，综合报告 "
        f"{int('research_report.md' in downloads)} 份。也可在下方分别下载 .md 文件。"
    )
    if available_count < len(result.papers):
        st.info("部分论文尚无可下载全文，压缩包仅包含已存在的文件；摘要总结不代表论文全文。")
    if result.warnings:
        with st.expander("运行提示与降级原因"):
            for warning in result.warnings:
                st.warning(warning)

    papers_tab, markdown_tab, summaries_tab, report_tab, benchmark_tab, audit_tab = st.tabs(
        ["输入论文（未排序）" if is_workbench else "Top-k 论文", "论文 Markdown",
         "单篇总结", "综合报告", "性能对比", "运行审计"]
    )

    with papers_tab:
        if is_workbench:
            st.caption("此入口仅处理指定论文，未执行相关性排序或代码仓库核验，因此不展示这些分数。")
        rows = []
        for rank, paper in enumerate(result.papers, 1):
            row = {
                "序号" if is_workbench else "排名": rank,
                "题目": paper.title,
                "年份": str(paper.year) if paper.year else "未提供",
                "发表位置": (paper.venue.name if paper.venue else None) or "未提供",
                "DOI": paper.doi or "未提供",
                "arXiv": paper.arxiv_id or "—",
                "开放获取": "是" if paper.is_open_access else "否" if paper.is_open_access is False else "未确认",
            }
            if not is_workbench:
                row.update({"代码状态": paper.code.status.value, "最终分数": paper.scores.final})
            rows.append(row)
        st.dataframe(rows, hide_index=True, use_container_width=True)
        for rank, paper in enumerate(result.papers, 1):
            with st.expander(f"{rank}. {paper.title}", expanded=rank == 1):
                metadata_received = any(
                    record.raw_metadata.get("metadata_method") == "arxiv_atom_api"
                    for record in paper.source_records
                )
                missing = "数据源未提供" if metadata_received else "本次未获取（可重新处理补全）"
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**作者**：{', '.join(a.name for a in paper.authors) or missing}")
                    institutions = ', '.join(paper.institutions) or (
                        "arXiv 未提供机构字段，不代表作者没有单位" if metadata_received else missing
                    )
                    st.markdown(f"**单位**：{institutions}")
                    st.markdown(f"**摘要**：{paper.abstract or missing}")
                with right:
                    sources = list(dict.fromkeys(
                        "arXiv" if r.source.value == "arxiv" else r.source.value
                        for r in paper.source_records
                    ))
                    source = '、'.join(sources) or (
                        "arXiv（按输入编号，基本信息未获取）" if paper.arxiv_id else missing
                    )
                    st.markdown(f"**来源**：{source}")
                    landing = paper.landing_page_url or paper.fulltext.landing_page_url
                    if landing:
                        st.link_button("查看论文来源", str(landing))
                    access = {
                        "open_access": "开放获取", "closed": "非开放获取",
                        "unavailable": "不可用", "unknown": "尚未确认",
                    }.get(paper.fulltext.access_status.value, "尚未确认")
                    st.markdown(f"**访问状态**：{access}")
                    parsed = paper_documents.get(paper.paper_id) is not None
                    st.markdown(f"**解析状态**：{'已生成全文 Markdown' if parsed else '尚无可用全文文件'}")
                    quality = paper.fulltext.parse_quality
                    st.markdown(f"**解析完整度**：{f'{quality:.2f}' if quality is not None else '未评估'}")
                    st.caption("完整度是文本覆盖与章节结构的启发式评分，不代表论文质量或总结正确性。")
                    if not is_workbench:
                        scores = paper.scores.model_dump(mode="json", exclude_none=True)
                        if scores:
                            with st.expander("查看排序分项"):
                                st.json(scores)
                        else:
                            st.caption("本次未获得排序分数。")

    with markdown_tab:
        st.caption("下载的是 PDF 解析得到的论文全文，不是模型改写或摘要扩写。")
        if not result.papers:
            st.info("本次没有可供下载的论文，请调整研究条件后重试。")
        for paper in result.papers:
            st.subheader(paper.title)
            markdown = paper_documents[paper.paper_id]
            st.download_button(
                "下载论文全文 Markdown", markdown or "",
                file_name=markdown_filename(paper.title, paper.paper_id, "paper"),
                mime="text/markdown", key=f"paper_{result.run_id}_{paper.paper_id}",
                disabled=markdown is None,
            )
            if markdown is None:
                st.info(
                    "暂无可下载的全文 Markdown：可能尚未解析、文件已失效或当前为演示数据。"
                    "可到“单篇总结”查看是否有基于摘要的总结。"
                )
            else:
                with st.expander("预览论文 Markdown"):
                    st.markdown(markdown[:20_000])
                    if len(markdown) > 20_000:
                        st.caption("预览仅显示前 20000 字符，下载文件包含完整正文。")
            st.divider()

    with summaries_tab:
        if not result.summaries:
            st.info("本次运行未生成单篇总结。")
        for summary in result.summaries:
            paper = paper_by_id.get(summary.paper_id)
            title = paper.title if paper else str(summary.paper_id)
            markdown = summary_documents[summary.paper_id]
            st.subheader(title)
            st.download_button(
                "下载单篇总结 Markdown", markdown,
                file_name=markdown_filename(title, summary.paper_id, "summary"),
                mime="text/markdown", key=f"summary_{result.run_id}_{summary.paper_id}",
            )
            summary_column, evidence_column = st.columns([3, 2])
            with summary_column:
                st.markdown(markdown)
            with evidence_column:
                st.markdown("#### 原文证据核对")
                st.caption("只核对引文文字，不代表结论或模型给出的页码已被验证。行号指向下载的 Markdown/摘要。")
                audit = evidence_audit(summary, summary_sources[summary.paper_id])
                matched = sum(row["核对"] == "原文匹配" for row in audit)
                st.caption(f"匹配 {matched}/{len(audit)} 条引文")
                if not audit:
                    st.info("该总结没有提供证据引用。")
                for row in audit:
                    label = f"引文 {row['序号']} · {row['章节']} · {row['核对']}"
                    with st.expander(label):
                        if row["起始行"]:
                            st.caption(f"材料第 {row['起始行']}–{row['结束行']} 行")
                            st.code(row["原文上下文"], language="text")
                        else:
                            st.warning("未能在现有材料中定位该引文，请人工复核。")
                        st.text(row["引文"])
            st.divider()

    with report_tab:
        st.caption("综合论文总结：基于本次已成功生成的单篇总结汇总，并提供跨论文对比。")
        if result.report and result.report.markdown.strip():
            st.download_button(
                "下载综合报告 Markdown", result.report.markdown,
                file_name="research_report.md", mime="text/markdown",
                key=f"report_{result.run_id}",
            )
            st.markdown(result.report.markdown)
        else:
            st.info("本次运行未生成综合报告。")

    with benchmark_tab:
        render_benchmark(result, settings)

    with audit_tab:
        st.dataframe(
            [event.model_dump(mode="json") for event in result.events],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "下载完整 JSON",
            result.model_dump_json(indent=2),
            file_name=f"pipeline_result_{result.run_id}.json",
            mime="application/json",
        )
else:
    st.info("先点击当前入口的处理按钮，生成后即可下载全文、单篇总结、综合报告，并查看证据与性能对比。")
