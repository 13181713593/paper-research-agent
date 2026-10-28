"""Browser discovery, candidate selection and user-chosen query context."""

import streamlit as st

from paper_agent.adapters.browser_search import ENGINES, BrowserPaperSearch, browser_search_url
from paper_agent.application.browser_search import BrowserSearchJob


@st.fragment(run_every=2)
def _running_browser_job(job):
    result, error, done = job.snapshot()
    if done:
        st.session_state.pop("browser_search_job", None)
        st.session_state["browser_search_result"] = result
        st.session_state["browser_search_error"] = error
        st.session_state.pop("browser_hit_selection", None)
        st.rerun()
    st.info("浏览器正在并行搜索；完成后可选择论文候选或用于检索词改写的内容。")


def render_browser_search(idea, *, disabled=False):
    job = st.session_state.get("browser_search_job")
    result = st.session_state.get("browser_search_result")
    with st.expander("浏览器论文检索与检索词辅助", expanded=bool(job or result)):
        st.caption(
            "使用独立浏览器检索学术网页。选中的论文可加入下一轮研究候选，"
            "标题、可取得的论文摘要或检索片段可辅助模型改写关键词。检索片段不等同于论文摘要。"
            "网页无相关论文时，使用同时查询的论文数据库补充，并分别标明来源。"
        )
        if "browser_query_text" not in st.session_state:
            st.session_state["browser_query_text"] = idea
        queries_text = st.text_area(
            "浏览器检索词（每行一条，最多4条，可同时填写中英文）",
            key="browser_query_text",
            disabled=bool(job) or disabled,
        )
        engines = st.multiselect(
            "搜索引擎",
            list(ENGINES),
            default=["Bing", "arXiv网页"],
            key="browser_engines",
            disabled=bool(job) or disabled,
        )
        c1, c2 = st.columns(2)
        if c1.button(
            "浏览器检索", disabled=bool(job) or disabled or not queries_text.strip() or not engines
        ):
            st.session_state["browser_search_job"] = BrowserSearchJob(
                BrowserPaperSearch(), queries_text.splitlines()[:4], engines
            ).start()
            st.session_state.pop("browser_search_error", None)
            st.rerun()
        if c2.button("使用当前研究主题", disabled=bool(job) or disabled):
            st.session_state["browser_query_pending"] = idea
            # Widget values are updated before their next instantiation.
            st.rerun()
        for engine in engines:
            if queries_text.strip():
                st.link_button(
                    f"在 {engine} 中查看", browser_search_url(engine, queries_text.splitlines()[0])
                )
        if job:
            _running_browser_job(job)
            return
        error = st.session_state.get("browser_search_error")
        if error:
            st.error(error)
        if result:
            with st.expander("查看本次实际网页检索地址"):
                for query in result.queries:
                    st.code(query, language=None)
            for warning in result.warnings:
                st.warning(warning)
            st.caption(
                f"找到 {len(result.hits)} 条网页结果 · 耗时 {result.duration_ms / 1000:.1f} 秒。"
                "论文日期以详情页元数据为准；日期未知时仅保留元数据，不凑新近文献配额。"
            )
            if result.hits:
                st.dataframe(
                    [
                        {
                            "序号": i + 1,
                            "题名": hit.title,
                            "来源": hit.engine,
                            "论文详情页": hit.paper_candidate,
                            "年份": hit.publication_year,
                            "作者": "、".join(hit.authors),
                            "元数据来源": hit.metadata_source or "未补全",
                            "链接": str(hit.url),
                            "检索片段": hit.snippet,
                        }
                        for i, hit in enumerate(result.hits)
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={"链接": st.column_config.LinkColumn("链接")},
                )
                selected = st.multiselect(
                    "选择要使用的结果",
                    range(len(result.hits)),
                    format_func=lambda i: f"{i + 1}. {result.hits[i].title}",
                    key="browser_hit_selection",
                    disabled=disabled,
                )
                selected_hits = [result.hits[i] for i in selected]
                c1, c2 = st.columns(2)
                if c1.button("加入论文候选", disabled=disabled or not selected_hits):
                    papers = {
                        str(p.landing_page_url): p
                        for p in st.session_state.get("browser_seed_papers", [])
                    }
                    for hit in selected_hits:
                        if hit.paper_candidate:
                            papers[str(hit.url)] = hit.to_paper()
                    st.session_state["browser_seed_papers"] = list(papers.values())[:30]
                    if any(not hit.paper_candidate for hit in selected_hits):
                        st.warning("非论文详情页的结果未加入候选，可用作检索词改写参考。")
                    st.success(
                        f"下一轮研究已准备 {len(st.session_state['browser_seed_papers'])} 篇浏览器论文候选。"
                    )
                if c2.button("用于辅助检索词改写", disabled=disabled or not selected_hits):
                    st.session_state["browser_search_context"] = [
                        hit.search_context()[:2000] for hit in selected_hits[:5]
                    ]
                    st.success(
                        "所选内容将在下一次“开始研究”时提供给查询规划模型；原研究主题保持不变。"
                    )
        seeds = st.session_state.get("browser_seed_papers", [])
        context = st.session_state.get("browser_search_context", [])
        if seeds or context:
            st.info(
                f"下一轮研究将使用：{len(seeds)} 篇浏览器论文候选、{len(context)} 段检索词改写参考。"
            )
            if st.button("清空下一轮补充内容", disabled=disabled):
                st.session_state.pop("browser_seed_papers", None)
                st.session_state.pop("browser_search_context", None)
                st.rerun()
