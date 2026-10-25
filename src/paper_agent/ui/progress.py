"""Live stage results and replay of persisted pipeline audit snapshots."""

from __future__ import annotations

import json

import streamlit as st

from paper_agent.domain.models import PipelineStage, ProcessingStatus, StageEvent

STAGE_LABELS = {
    PipelineStage.QUERY_PLANNED: "查询规划",
    PipelineStage.RETRIEVED: "多源数据库检索",
    PipelineStage.DEDUPED: "合并去重",
    PipelineStage.PRE_RANKED: "元数据预排序",
    PipelineStage.FULLTEXT_PROCESSED: "全文处理",
    PipelineStage.VERIFIED: "证据确认",
    PipelineStage.FINAL_RANKED: "最终重排",
    PipelineStage.SUMMARIZED: "单篇总结",
    PipelineStage.REPORTED: "综合报告",
}
SOURCE_LABELS = {
    "arxiv": "arXiv",
    "crossref": "Crossref",
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "europe_pmc": "Europe PMC",
}
STATUS_LABELS = {
    ProcessingStatus.RUNNING: "⏳ 处理中",
    ProcessingStatus.SUCCEEDED: "✅ 已完成",
    ProcessingStatus.PARTIAL: "⚠️ 已完成，存在降级或警告",
    ProcessingStatus.FAILED: "❌ 失败",
    ProcessingStatus.SKIPPED: "⏭️ 已排除",
}


def is_legacy_empty_result(error):
    """Recognize old stored arXiv empty-query exceptions without hiding real failures."""
    return "arXiv query returned no results:" in (error or "")


def query_result_row(query):
    empty = (query["status"] == "succeeded" and query["count"] == 0) or is_legacy_empty_result(
        query.get("error")
    )
    return {
        "检索式": query["query"],
        "语言": query.get("language"),
        "时间窗口": query.get("time_window"),
        "日期范围": f"{query.get('date_from') or '不限'} 至 {query.get('date_to') or '不限'}",
        "排序": "发表时间倒序" if query.get("sort_by") == "publication_date" else "相关性",
        "状态": "未匹配"
        if empty
        else {"succeeded": "查询成功", "failed": "请求失败"}.get(query["status"], query["status"]),
        "返回数量": query["count"],
        "说明": "查询已返回，未找到匹配文献" if empty else query.get("error"),
    }


def paper_table(papers: list[dict], stage: PipelineStage) -> None:
    if not papers:
        st.info("本阶段没有输出文献。")
        return
    rows = []
    for rank, paper in enumerate(papers, 1):
        scores = paper.get("scores") or {}
        fulltext = paper.get("fulltext") or {}
        row = {
            "序号": rank,
            "文献题目": paper["title"],
            "作者": ", ".join(paper.get("authors", [])),
            "年份": paper.get("year"),
            "来源数据库": ", ".join(SOURCE_LABELS.get(s, s) for s in paper.get("sources", [])),
            "DOI": paper.get("doi"),
            "arXiv ID": paper.get("arxiv_id"),
            "文献链接": paper.get("url"),
        }
        if stage in {PipelineStage.PRE_RANKED, PipelineStage.FINAL_RANKED, PipelineStage.VERIFIED}:
            row.update(
                {
                    "分数": scores.get("final"),
                    "证据覆盖度": scores.get("evidence_coverage"),
                    "评分解释": scores.get("explanation"),
                }
            )
        if stage == PipelineStage.FULLTEXT_PROCESSED:
            row.update(
                {
                    "访问状态": fulltext.get("access_status"),
                    "解析器": fulltext.get("parser_name"),
                    "解析质量": fulltext.get("parse_quality"),
                    "解析路径": fulltext.get("markdown_path"),
                }
            )
        row["摘要"] = paper.get("abstract")
        rows.append(row)
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={"文献链接": st.column_config.LinkColumn("文献链接")},
    )


def _select_stage(key, value):
    st.session_state[key] = value


class PipelineProgress:
    def __init__(self, stages: list[PipelineStage], *, key="pipeline"):
        self.stages = stages or list(STAGE_LABELS)
        self.latest: dict[PipelineStage, StageEvent] = {}
        self.children: dict[PipelineStage, dict[str, StageEvent]] = {s: {} for s in self.stages}
        self.finished = False
        self.status = None
        self.key = key

    def update(self, event: StageEvent, *, render: bool = False) -> None:
        if event.stage not in self.stages:
            return
        if event.scope == "stage":
            self.latest[event.stage] = event
        else:
            child_key = event.source_name if event.scope == "source" else str(event.paper_id)
            self.children[event.stage][child_key] = event
        if render:
            self.render()

    def replay(self, events: list[StageEvent], status: ProcessingStatus | None = None) -> None:
        for event in events:
            self.update(event)
        self.finished = status is not None
        self.status = status
        self.render()

    def render(self) -> None:
        completed = sum(
            e.status in {ProcessingStatus.SUCCEEDED, ProcessingStatus.PARTIAL}
            for e in self.latest.values()
        )
        ended = sum(e.status != ProcessingStatus.RUNNING for e in self.latest.values())
        running = next(
            (
                s
                for s in self.stages
                if s in self.latest and self.latest[s].status == ProcessingStatus.RUNNING
            ),
            None,
        )
        pending = len(self.stages) - len(self.latest)
        current = running or next(
            (s for s in reversed(self.stages) if s in self.latest), self.stages[0]
        )
        selection_key = f"{self.key}_selected_stage"
        selected_value = st.session_state.get(selection_key)
        selected = next(
            (s for s in self.stages if s.value == selected_value and s in self.latest), current
        )
        labels = {
            ProcessingStatus.SUCCEEDED: "✅ 已完成",
            ProcessingStatus.PARTIAL: "⚠️ 已完成 · 有警告",
            ProcessingStatus.RUNNING: "🔵 进行中",
            ProcessingStatus.FAILED: "❌ 失败",
            ProcessingStatus.SKIPPED: "⏭️ 已跳过",
        }
        st.markdown(
            """
        <style>
        .st-key-pipeline_navigation button {min-height: 64px; white-space: pre-line;
          text-align: left; border-radius: 12px;}
        .st-key-pipeline_navigation button p {white-space: pre-line; font-size: .88rem;}
        .st-key-pipeline_navigation button[kind="primary"] {background: #2563eb;
          border-color: #2563eb; color: white;}
        .st-key-pipeline_navigation button:disabled {opacity: .5;}
        .st-key-pipeline_follow_control button {min-height: 36px;}
        </style>
        """,
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="pipeline_navigation"):
            st.subheader("研究进度")
            if self.finished:
                title = {
                    ProcessingStatus.FAILED: "流程已停止，请查看失败阶段",
                    ProcessingStatus.PARTIAL: "处理完成，部分来源或文献有警告",
                }.get(self.status, "全部阶段已完成")
            else:
                title = f"正在进行：{STAGE_LABELS[current]}" if running else "等待开始研究"
            st.markdown(f"**{title}**")
            pending_label = "未执行" if self.finished else "待执行"
            st.progress(
                ended / len(self.stages),
                text=f"已完成 {completed} / {len(self.stages)} · 进行中 {int(running is not None)} · {pending_label} {pending}",
            )
            st.caption("点击已完成或进行中的阶段查看详情；待执行阶段会在开始后开放。")
            for offset in range(0, len(self.stages), 3):
                columns = st.columns(3, gap="small")
                for column, name in zip(columns, self.stages[offset : offset + 3], strict=False):
                    event = self.latest.get(name)
                    state = labels[event.status] if event else f"○ {pending_label}"
                    with column:
                        st.button(
                            f"{self.stages.index(name) + 1:02d}  {STAGE_LABELS[name]}\n{state}",
                            key=f"{self.key}_stage_{name.value}",
                            disabled=event is None,
                            type="primary" if name == selected and event else "secondary",
                            width="stretch",
                            on_click=_select_stage,
                            args=(selection_key, name.value),
                        )
            with st.container(key="pipeline_follow_control"):
                st.button(
                    "跟随当前进度" if not self.finished else "查看最后阶段",
                    key=f"{self.key}_follow",
                    disabled=selected_value is None,
                    on_click=_select_stage,
                    args=(selection_key, None),
                )
        if not self.latest:
            return
        with st.container(border=True, key="pipeline_stage_detail"):
            st.subheader(f"{self.stages.index(selected) + 1:02d} · {STAGE_LABELS[selected]}")
            if selected_value is None and not self.finished:
                st.caption("正在跟随当前进度；点击上方阶段可固定查看。")
            self.render_stage(selected)

    def render_stage(self, name: PipelineStage) -> None:
        event = self.latest.get(name)
        with st.container():
            if event is None:
                st.caption("未执行（流程已提前结束）" if self.finished else "等待执行")
                return
            details = event.details
            timing = (
                f" · 耗时 {event.duration_ms / 1000:.2f} 秒"
                if event.duration_ms is not None
                else ""
            )
            st.markdown(f"**{STATUS_LABELS.get(event.status, event.status.value)}**{timing}")
            if details.get("demo"):
                st.caption("此阶段使用演示组件，输出仅用于流程联调。")
            if event.input_count is not None or event.output_count is not None:
                st.caption(
                    f"输入：{event.input_count if event.input_count is not None else '—'} 条 · "
                    f"输出：{event.output_count if event.output_count is not None else '—'} 条"
                )
            if event.message:
                st.error(event.message)
            for warning in details.get("warnings", []):
                if is_legacy_empty_result(warning):
                    st.info(
                        "历史记录中 arXiv 的空结果曾被标为失败；这表示该检索式未匹配，已找到的文献仍然有效。"
                    )
                else:
                    st.warning(warning)

            if name == PipelineStage.QUERY_PLANNED and "plan" in details:
                plan = details["plan"]
                for warning in plan.get("warnings", []):
                    st.warning(warning)
                st.write("研究意图：", plan["intent_summary"])
                if plan.get("recent_from"):
                    st.info(
                        f"检索范围：{plan['date_from']} 至 {plan['date_to']}；"
                        f"优先 {plan['recent_from']} 之后的文献，近2年目标占比至少 {plan.get('recent_share', 0.7):.0%}（相关候选充足时）。"
                    )
                st.dataframe(
                    [
                        {
                            "数据库": SOURCE_LABELS.get(q["source"], q["source"]),
                            "语言": q.get("language") or "默认",
                            "时间窗口": q.get("time_window"),
                            "排序": "发表时间倒序"
                            if q.get("sort_by") == "publication_date"
                            else "相关性",
                            "检索策略": q.get("strategy") or "默认",
                            "检索式": q["query"],
                            "最多召回": q["limit"],
                            "过滤条件": json.dumps(q.get("filters", {}), ensure_ascii=False),
                        }
                        for q in plan["queries"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.write("候选同义词：", "、".join(plan.get("synonyms", [])) or "无")
                if plan.get("query_terms"):
                    st.write("本轮实际检索用词：", "；".join(plan["query_terms"]))
                    st.caption(
                        "每条表达分别请求数据库，结果合并去重；最多使用6条英文表达及中文原始主题。"
                        "请求是否成功、各词返回数量请查看“多源数据库检索”阶段。"
                    )
                if plan.get("term_notes"):
                    with st.expander("查看扩展词的采用与改写说明"):
                        st.dataframe(
                            [
                                {"候选词": term, "处理说明": note}
                                for term, note in plan["term_notes"].items()
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                st.write("纳入条件：", "；".join(plan.get("inclusion_criteria", [])) or "无")
                st.write("排除条件：", "；".join(plan.get("exclusion_criteria", [])) or "无")
                if plan.get("planner_mode"):
                    st.caption(
                        "查询改写方式："
                        + (
                            "大模型改写"
                            if plan["planner_mode"] == "llm"
                            else "本地双语词典与规则（模型未启用或调用失败）"
                        )
                    )

            children = list(self.children[name].values())
            if name == PipelineStage.RETRIEVED:
                done = sum(c.status != ProcessingStatus.RUNNING for c in children)
                if children:
                    st.caption(
                        f"数据库进度：{done} / {len(children)} · "
                        "每个数据库完成后立即显示结果，其他来源继续检索。"
                    )
                for source in children:
                    with st.container(border=True):
                        source_label = SOURCE_LABELS.get(source.source_name, source.source_name)
                        attempts = source.details.get("query_results", [])
                        legacy_empty_only = (
                            source.status != ProcessingStatus.RUNNING
                            and any(is_legacy_empty_result(q.get("error")) for q in attempts)
                            and all(
                                q["status"] == "succeeded" or is_legacy_empty_result(q.get("error"))
                                for q in attempts
                            )
                        )
                        source_status = (
                            ProcessingStatus.SUCCEEDED if legacy_empty_only else source.status
                        )
                        st.markdown(f"**{source_label}** · {STATUS_LABELS[source_status]}")
                        if source.details.get("demo"):
                            st.caption("演示数据库适配器，以下为示例文献。")
                        for query in source.details.get("queries", []):
                            st.code(query["query"], language=None)
                        if source.duration_ms is not None:
                            st.caption(f"耗时 {source.duration_ms / 1000:.2f} 秒")
                        if source.details.get("recency"):
                            recency = source.details["recency"]
                            st.caption(
                                f"近2年文献：{recency['recent_count']} / {recency['total']}；"
                                f"优先日期：{recency['recent_from']} 起，目标占比 {recency['target_share']:.0%}。"
                            )
                        if source.message:
                            if is_legacy_empty_result(source.message):
                                st.caption(
                                    "部分检索式未匹配文献，属于正常空结果；已召回文献仍保留。"
                                )
                            else:
                                st.error(f"{source.error_code or '检索失败'}：{source.message}")
                        if source.details.get("query_results"):
                            st.dataframe(
                                [query_result_row(q) for q in source.details["query_results"]],
                                hide_index=True,
                                width="stretch",
                            )
                        if (
                            source.status
                            in {
                                ProcessingStatus.SUCCEEDED,
                                ProcessingStatus.PARTIAL,
                                ProcessingStatus.RUNNING,
                            }
                            and source.output_count is not None
                        ):
                            st.write(f"检索到 {source.output_count} 篇文献")
                            paper_table(source.details.get("papers", []), name)
                if not children and "papers" in details:
                    paper_table(details["papers"], name)
                return

            if children:
                done = sum(c.status != ProcessingStatus.RUNNING for c in children)
                st.caption(f"文献处理进度：{done} / {event.input_count or len(children)}")
                st.dataframe(
                    [
                        {
                            "文献题目": c.details.get("title"),
                            "状态": STATUS_LABELS[c.status],
                            "说明": c.message,
                            "耗时（秒）": c.duration_ms / 1000
                            if c.duration_ms is not None
                            else None,
                        }
                        for c in children
                    ],
                    hide_index=True,
                    width="stretch",
                )
                if name == PipelineStage.FULLTEXT_PROCESSED:
                    attempts = [
                        {"文献": c.details.get("title"), **attempt}
                        for c in children
                        for p in c.details.get("papers", [])
                        for attempt in p.get("fulltext", {}).get("attempts", [])
                    ]
                    if attempts:
                        st.caption("公开全文获取尝试记录")
                        st.dataframe(attempts, hide_index=True, width="stretch")

            if "papers" in details:
                if name == PipelineStage.DEDUPED:
                    st.write(
                        f"合并去重：{event.input_count} → {event.output_count} 篇，"
                        f"减少 {(event.input_count or 0) - (event.output_count or 0)} 条重复记录。"
                    )
                paper_table(details["papers"], name)
            elif children and name != PipelineStage.SUMMARIZED:
                completed_papers = [p for c in children for p in c.details.get("papers", [])]
                if completed_papers:
                    paper_table(completed_papers, name)

            summaries = details.get("summaries") or [
                summary for c in children for summary in c.details.get("summaries", [])
            ]
            titles = {str(c.paper_id): c.details.get("title") for c in children}
            for summary in summaries:
                st.markdown(f"**{titles.get(summary['paper_id']) or summary['paper_id']}**")
                st.write("核心问题：", summary["core_problem"])
                st.write("方法：", summary["method"])
                st.write("实验：", summary["experiments"])
                st.write("贡献：", "；".join(summary["contributions"]))
                st.write("局限：", "；".join(summary["limitations"]))
                st.caption("全文总结" if summary["generated_from_fulltext"] else "摘要降级总结")
            if "report" in details:
                st.markdown(details["report"]["markdown"])
