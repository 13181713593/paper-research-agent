"""Streamlit UI for the shared main-agent facade."""

from __future__ import annotations

import os

import streamlit as st

from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import PipelineRequest, ProcessingStatus, ResearchIdea
from paper_agent.domain.publication_policy import is_recent
from paper_agent.ui.progress import HIDDEN_STAGES, PipelineProgress
from paper_agent.ui.research_job import ResearchJob
from paper_agent.ui.stage_panels import render_home_download, render_release_results
from paper_agent.ui.workbench import render_processing_metrics, render_workbench

st.set_page_config(page_title="论文研究智能体", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 5rem; padding-bottom: 4rem;}
    [data-testid="stAppViewContainer"] {
      background: radial-gradient(ellipse at top right, #eaf2ff55, transparent 55%);
    }
    [data-testid="stSidebar"] {border-right: 1px solid #94a3b826;}
    h1 {letter-spacing: -.035em; font-weight: 800 !important;}
    h3 {letter-spacing: -.02em;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
    .stButton button, .stDownloadButton button {border-radius: 10px; min-height: 44px;}
    .stButton button[kind="primary"] {
      background: linear-gradient(120deg, #2563eb, #4f46e5); border: 0;
      box-shadow: 0 4px 14px #2563eb24;
    }
    .research-hero {padding: 1.8rem 2rem; border-radius: 22px; margin-bottom: 1.5rem;
      background: linear-gradient(120deg, #132b50, #234ca0 65%, #4f46a1); color: #fff;}
    .research-hero .eyebrow {color: #b8d6ff; font-size: .8rem; letter-spacing: .14em;}
    .research-hero h1 {color: white; font-size: clamp(1.8rem, 3vw, 2.65rem); padding: .6rem 0;}
    .research-hero p {color: #e0eaff; margin-bottom: .3rem; line-height: 1.8;}
    @media (max-width: 640px) {
      .block-container {padding-top: 5rem;} .research-hero {padding: 1.3rem;}
    }
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent(settings: Settings):
    """Build the agent with source-level publication windows and newest-first retrieval."""
    return build_agent(settings)


def store_research_result(result):
    st.session_state["pipeline_result"] = result
    st.session_state[f"pipeline_{result.run_id}_selected_stage"] = st.session_state.get(
        "pipeline_live_selected_stage"
    )
    st.session_state.pop("research_job", None)
    st.session_state.pop("processing_run", None)
    st.session_state.pop("processing_benchmark", None)


@st.fragment(run_every=1)
def render_running_research(job, stages, debug):
    events, result, error, done = job.snapshot()
    if done and result is not None:
        store_research_result(result)
        st.rerun()
    if error and debug:
        st.error(error)
    st.caption(f"当前任务：{job.request.idea.text}")
    PipelineProgress(stages, key="pipeline_live", compact=not debug).replay(
        events, ProcessingStatus.FAILED if error else None
    )


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in value.replace("，", ",").split(",") if term.strip()]


st.markdown(
    '<div class="research-hero"><div class="eyebrow">PAPER RESEARCH · 文献研究工作台</div>'
    "<h1>从研究想法，到有据可循的综述</h1>"
    "<p>多源发现相关论文，整理全文与关键证据，让研究线索汇成清晰的报告。</p>"
    "<p>多源检索 &nbsp; / &nbsp; 全文阅读 &nbsp; / &nbsp; 研究综述</p></div>",
    unsafe_allow_html=True,
)

load_local_env()
settings = Settings.from_env()
if settings.app_env in {"demo", "ranking"}:
    st.warning(
        "当前使用演示检索：论文为虚构示例，不是真实检索结果。"
        "请将 PAPER_AGENT_ENV 改为 production，重启服务并重新生成结果。"
    )
job = st.session_state.get("research_job")
research_running = job is not None and not job.snapshot()[3]
mode = "完整研究流程"

with st.sidebar:
    st.subheader("显示模式")
    display_mode = st.radio(
        "页面显示",
        ["运行模式", "调试模式"],
        key="display_mode",
        help="运行模式聚焦当前进度与研究成果；调试模式展示每个阶段的详细结果。",
    )
    debug = display_mode == "调试模式"
    st.divider()
    st.header("任务配置")
    top_k = st.slider("最终论文数 Top-k", 1, 100, 5, help="全文确认后最终返回的论文数")
    pool_size = st.slider(
        "全文候选池 N",
        top_k,
        200,
        max(20, top_k),
        help="预排序后进入下载解析的候选数；N 应明显大于 k，以便失败补位。",
    )
    source_limit = st.slider("每个来源最多召回", 5, 200, 30)
    st.caption(
        "默认检索近5年文献；近2年相关文献优先，目标占比至少70%。各来源采用日期筛选与适配的排序方式。"
    )
    st.number_input("当前下载 / 总结并发上限", 1, 32, settings.max_concurrency, disabled=True)
    st.caption("批量数量与并发数独立；增大候选池不会同时发出同等数量的下载请求。")
    require_fulltext = st.checkbox("仅保留成功解析全文", value=False)
    allow_abstract_fallback = st.checkbox("全文失败时允许摘要降级", value=True)
    st.divider()
    contract = get_agent(settings).contract()
    st.caption(f"服务版本 {contract.version} · 至少 {contract.minimum_sources} 个数据源")
    independent = mode == "第三组独立演示"
    if independent:
        st.info("独立入口：按 arXiv 编号获取元数据并真实下载、解析和总结；不执行检索或排序。")
    else:
        mode_descriptions = {
            "demo": "demo：全部使用离线演示组件，不执行真实检索或全文下载。",
            "retrieval": "retrieval：六源真实检索与去重；排序、全文处理、总结和报告仍为演示组件。",
            "ranking": "ranking：第二组真实预排序、证据确认和重排；检索、全文处理和总结仍为演示组件。",
            "production": "production：六源真实检索 → 去重 → 第二组排序 → 第三组全文处理与总结报告。",
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
        if not (
            (os.getenv("OPENAI_API_KEY") or "").strip()
            or (os.getenv("PAPER_AGENT_LLM_API_KEY") or "").strip()
        ):
            st.info("未配置总结 API：单篇总结使用抽取式降级，综述使用模板并注明证据限制。")
        else:
            st.caption(
                f"已配置总结模型：{settings.llm_model}。"
                "生成时会调用 API；调用失败时会降级为抽取式总结与模板报告。"
            )

idea = st.text_area(
    "研究想法",
    value="",
    key="research_idea",
    height=130,
    placeholder="请描述研究对象、核心问题、方法侧重点和期望时间范围……",
    help="支持简短研究主题，例如“场景文本检测与识别”或“OCR”；补充方法与时间范围可帮助筛选。",
)
idea_text = idea.strip()

with st.expander("可选检索约束"):
    c1, c2, c3 = st.columns(3)
    required_text = c1.text_input("必须包含词（逗号分隔）")
    excluded_text = c2.text_input("排除词（逗号分隔）")
    focus_text = c3.text_input("排序侧重点（逗号分隔）", placeholder="方法相似度, 最新工作")

if mode == "第三组独立演示":
    render_workbench(settings, idea, debug=debug)
else:
    st.caption("生成论文清单、单篇总结与中文综合报告，并统一下载研究资料。")
start_research = mode == "完整研究流程" and st.button(
    "开始研究（下载论文并生成总结）",
    type="primary",
    disabled=not idea_text or research_running,
    use_container_width=True,
)
# An edit and a click can arrive in the same rerun, before the browser sees disabled=True.
start_research = start_research and bool(idea_text) and not research_running
if start_research:
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("research_bundle", None)
    request = PipelineRequest(
        idea=ResearchIdea(
            text=idea_text,
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
    st.session_state["pipeline_live_selected_stage"] = None
    st.session_state["research_job"] = ResearchJob(get_agent(settings), request).start()
    if not st.session_state["research_job"].snapshot()[3]:
        st.rerun()

job = st.session_state.get("research_job")
if job is not None:
    _, completed_result, _, done = job.snapshot()
    if done and completed_result is not None:
        store_research_result(completed_result)
    else:
        render_running_research(job, contract.stages, debug)

result = st.session_state.get("pipeline_result")
if result:
    if debug:
        render_processing_metrics(result)
    processing_run = st.session_state.get("processing_run")
    is_workbench = bool(processing_run and processing_run.result.run_id == result.run_id)
    # 下载、预览与审计随各阶段渲染，不再集中到最后一段统一总结。
    PipelineProgress(
        contract.stages,
        key=f"pipeline_{result.run_id}",
        result=result,
        settings=settings,
        workbench=is_workbench,
        compact=not debug,
    ).replay(result.progress_events or result.events, result.status)
    render_home_download(result, settings)
    if not debug:
        render_release_results(result)
    if debug:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最终论文", len(result.papers))
        m2.metric(
            "完成阶段",
            sum(
                e.status != ProcessingStatus.FAILED
                for e in result.events
                if e.stage not in HIDDEN_STAGES
            ),
        )
        m3.metric("累计阶段耗时", f"{result.total_duration_ms / 1000:.2f}s")
        m4.metric("警告", len(result.warnings))
    plan_data = next(
        (e.details.get("plan", {}) for e in result.events if e.stage.value == "query_planned"), {}
    )
    if plan_data.get("recent_from"):
        from datetime import date

        recent_cutoff = date.fromisoformat(plan_data["recent_from"])
        recent_count = sum(is_recent(p, recent_cutoff) for p in result.papers)
        st.caption(
            f"本轮检索范围：{plan_data['date_from']} 至 {plan_data['date_to']}；"
            f"最终结果中近2年文献 {recent_count}/{len(result.papers)} 篇。"
            "若相关新文献不足，会在近5年范围内补充，不为凑比例纳入无关文献。"
        )

elif st.session_state.get("research_job") is None:
    if mode == "完整研究流程":
        PipelineProgress(contract.stages, key="pipeline_idle", compact=not debug).replay([])
    else:
        st.info("先处理论文，完成后可下载全文、单篇总结和综合报告。")
