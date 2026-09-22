"""Streamlit UI for the shared main-agent facade."""

from __future__ import annotations

import asyncio

import streamlit as st

from paper_agent.bootstrap import build_agent
from paper_agent.domain.models import PipelineRequest, ProcessingStatus, ResearchIdea

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

if st.button("开始研究", type="primary", disabled=len(idea.strip()) < 10, use_container_width=True):
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

result = st.session_state.get("pipeline_result")
if result:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最终论文", len(result.papers))
    m2.metric("完成阶段", len(result.events))
    m3.metric("累计阶段耗时", f"{result.total_duration_ms / 1000:.2f}s")
    m4.metric("警告", len(result.warnings))

    for warning in result.warnings:
        st.warning(warning)

    papers_tab, summaries_tab, report_tab, audit_tab = st.tabs(
        ["Top-k 论文", "单篇总结", "综合报告", "运行审计"]
    )

    with papers_tab:
        rows = []
        for rank, paper in enumerate(result.papers, 1):
            rows.append(
                {
                    "排名": rank,
                    "题目": paper.title,
                    "年份": paper.year,
                    "发表位置": paper.venue.name if paper.venue else None,
                    "DOI": paper.doi,
                    "arXiv": paper.arxiv_id,
                    "开放获取": paper.is_open_access,
                    "代码状态": paper.code.status.value,
                    "最终分数": paper.scores.final,
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        for rank, paper in enumerate(result.papers, 1):
            with st.expander(f"{rank}. {paper.title}", expanded=rank == 1):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**作者**：{', '.join(a.name for a in paper.authors) or '未知'}")
                    st.markdown(f"**单位**：{', '.join(paper.institutions) or '未知'}")
                    st.markdown(f"**摘要**：{paper.abstract or '无摘要'}")
                with right:
                    st.markdown(f"**来源**：{', '.join(r.source.value for r in paper.source_records)}")
                    st.markdown(f"**全文状态**：{paper.fulltext.access_status.value}")
                    st.markdown(f"**解析质量**：{paper.fulltext.parse_quality or 0:.2f}")
                    st.json(paper.scores.model_dump(mode="json"))

    with summaries_tab:
        paper_by_id = {paper.paper_id: paper for paper in result.papers}
        for summary in result.summaries:
            paper = paper_by_id.get(summary.paper_id)
            st.subheader(paper.title if paper else str(summary.paper_id))
            st.markdown(f"**核心问题**：{summary.core_problem}")
            st.markdown(f"**方法**：{summary.method}")
            st.markdown(f"**实验**：{summary.experiments}")
            st.markdown("**贡献**：" + "；".join(summary.contributions))
            st.markdown("**局限**：" + "；".join(summary.limitations))
            st.caption("基于全文" if summary.generated_from_fulltext else "仅基于摘要降级生成")
            st.divider()

    with report_tab:
        if result.report:
            st.markdown(result.report.markdown)
            st.download_button(
                "下载 Markdown 报告", result.report.markdown,
                file_name="research_report.md", mime="text/markdown",
            )
        else:
            st.info("本次运行未生成报告。")

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
