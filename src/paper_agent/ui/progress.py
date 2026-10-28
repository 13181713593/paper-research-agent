"""Live stage results and replay of persisted pipeline audit snapshots."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import urlparse

import streamlit as st

from paper_agent.application.progress import redact_audit_value
from paper_agent.domain.models import PipelineStage, ProcessingStatus, StageEvent
from paper_agent.ui import stage_panels

STAGE_LABELS = {
    PipelineStage.QUERY_PLANNED: "查询规划",
    PipelineStage.RETRIEVED: "多源数据库检索",
    PipelineStage.DEDUPED: "合并去重",
    PipelineStage.PRE_RANKED: "元数据预排序",
    PipelineStage.DOWNLOADED: "论文下载",
    PipelineStage.MARKDOWN_CONVERTED: "转换为 Markdown",
    PipelineStage.FINAL_RANKED: "最终重排",
    PipelineStage.SUMMARIZED: "单篇总结",
    PipelineStage.REPORTED: "综合报告",
    # 历史运行：分段化之前的“全文处理”阶段，回放时按下载阶段展示。
    PipelineStage.FULLTEXT_PROCESSED: "全文处理（历史）",
    PipelineStage.VERIFIED: "证据确认",
}
DEFAULT_STAGES = [
    PipelineStage.QUERY_PLANNED,
    PipelineStage.RETRIEVED,
    PipelineStage.DEDUPED,
    PipelineStage.PRE_RANKED,
    PipelineStage.DOWNLOADED,
    PipelineStage.MARKDOWN_CONVERTED,
    PipelineStage.FINAL_RANKED,
    PipelineStage.SUMMARIZED,
    PipelineStage.REPORTED,
]
ALIASED_STAGES = {PipelineStage.FULLTEXT_PROCESSED: PipelineStage.DOWNLOADED}
# 证据确认不再是独立阶段，其明细并入“最终重排”展示。
HIDDEN_STAGES = {PipelineStage.VERIFIED}
CHANNEL_LABELS = {
    "direct": "直链",
    "unpaywall": "Unpaywall",
    "openalex": "OpenAlex",
    "landing_page": "落地页",
    "resolve": "学术库",
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


def paper_fulltext(event: StageEvent) -> dict:
    papers = event.details.get("papers") or []
    return (papers[0].get("fulltext") or {}) if papers else {}


def download_source(fulltext: dict) -> tuple[str | None, str]:
    """说明该篇全文实际来自哪个渠道：返回（渠道名, 主机名）。"""
    for attempt in reversed(fulltext.get("attempts") or []):
        if attempt.get("stage") == "download" and attempt.get("status") == "succeeded":
            channel = CHANNEL_LABELS.get(attempt.get("channel") or "", "直链")
            host = urlparse(attempt.get("url") or "").hostname or ""
            return channel, host
    if fulltext.get("markdown_path"):
        return "本地缓存", ""
    return None, ""


def download_row(index: int, event: StageEvent) -> dict:
    fulltext = paper_fulltext(event)
    found = bool(fulltext.get("local_pdf_path") or fulltext.get("markdown_path"))
    channel, host = download_source(fulltext)
    return {
        "序号": index,
        "文献题目": event.details.get("title"),
        "是否找到": "✅ 找到公开全文" if found else "❌ 未找到",
        "取得来源": f"{channel}（{host}）" if channel and host else (channel or "—"),
        "本地文件": "已下载 PDF"
        if fulltext.get("local_pdf_path")
        else ("复用已解析 Markdown" if fulltext.get("markdown_path") else "—"),
        "访问状态": fulltext.get("access_status"),
        "耗时（秒）": event.duration_ms / 1000 if event.duration_ms is not None else None,
        "说明": event.message,
    }


def conversion_row(index: int, event: StageEvent) -> dict:
    fulltext = paper_fulltext(event)
    if event.status == ProcessingStatus.SKIPPED:
        state = "⏭️ 跳过（未取得 PDF）"
    elif event.status == ProcessingStatus.RUNNING:
        state = "⏳ 转换中"
    elif event.status in {ProcessingStatus.SUCCEEDED, ProcessingStatus.PARTIAL} and fulltext.get(
        "markdown_path"
    ):
        state = "✅ 已转换"
    else:
        state = "❌ 转换失败"
    return {
        "序号": index,
        "文献题目": event.details.get("title"),
        "转换状态": state,
        "解析器": fulltext.get("parser_name"),
        "解析质量": fulltext.get("parse_quality"),
        "Markdown 文件": fulltext.get("markdown_path"),
        "耗时（秒）": event.duration_ms / 1000 if event.duration_ms is not None else None,
        "说明": event.message,
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


def fulltext_attempt_rows(children: list[StageEvent]) -> list[dict]:
    """导出整个已处理候选池，不受最终 Top-k 数量影响。"""
    return redact_audit_value(
        [
            {
                "文献": child.details.get("title"),
                "paper_id": str(child.paper_id),
                "DOI": paper.get("doi"),
                "序号": index,
                **attempt,
            }
            for child in children
            for paper in child.details.get("papers", [])
            for index, attempt in enumerate(paper.get("fulltext", {}).get("attempts", []), 1)
        ]
    )


def fulltext_attempt_csv(rows: list[dict]) -> bytes:
    """UTF-8 BOM 兼容中文电子表格，并防止外部文本触发公式。"""
    output = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in redact_audit_value(rows):
        writer.writerow(
            {
                key: "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8-sig")


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
                    "分数（满分2）": scores.get(
                        "display_score",
                        2 * scores["final"] if scores.get("final") is not None else None,
                    ),
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
    def __init__(
        self,
        stages: list[PipelineStage],
        *,
        key="pipeline",
        result=None,
        settings=None,
        workbench: bool = False,
    ):
        self.stages = stages or list(DEFAULT_STAGES)
        self.hidden = [stage for stage in HIDDEN_STAGES if stage not in self.stages]
        self.latest: dict[PipelineStage, StageEvent] = {}
        self.children: dict[PipelineStage, dict[str, StageEvent]] = {s: {} for s in self.stages}
        self.hidden_events: dict[PipelineStage, dict[str, StageEvent]] = {
            stage: {} for stage in self.hidden
        }
        self.finished = False
        self.status = None
        self.key = key
        # 已完成的运行会带上结果与配置，各阶段据此就地渲染下载与预览。
        self.result = result
        self.settings = settings
        self.workbench = workbench

    @staticmethod
    def _child_key(event: StageEvent) -> str:
        if event.scope == "stage":
            return "__stage__"
        return event.source_name if event.scope == "source" else str(event.paper_id)

    def update(self, event: StageEvent, *, render: bool = False) -> None:
        stage = ALIASED_STAGES.get(event.stage, event.stage)
        if stage in self.stages:
            if event.scope == "stage":
                self.latest[stage] = event
            else:
                self.children[stage][self._child_key(event)] = event
        elif stage in self.hidden_events:
            self.hidden_events[stage][self._child_key(event)] = event
        else:
            return
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

            if name == PipelineStage.DOWNLOADED:
                self.render_download(event)
                self.render_result_panel(name)
                return
            if name == PipelineStage.MARKDOWN_CONVERTED:
                self.render_conversion(event)
                self.render_result_panel(name)
                return

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
                    # 历史记录回放才会走到这里：旧运行没有分段事件。
                    attempts = fulltext_attempt_rows(children)
                    if attempts:
                        st.caption("公开全文获取尝试记录（查询成功不等于 PDF 下载成功）")
                        st.dataframe(attempts, hide_index=True, width="stretch")

            if "papers" in details:
                if name == PipelineStage.DEDUPED:
                    st.write(
                        f"合并去重：{event.input_count} → {event.output_count} 篇，"
                        f"减少 {(event.input_count or 0) - (event.output_count or 0)} 条重复记录。"
                    )
                if self.result is None or name != PipelineStage.FINAL_RANKED:
                    paper_table(details["papers"], name)
            elif children and name != PipelineStage.SUMMARIZED:
                completed_papers = [p for c in children for p in c.details.get("papers", [])]
                if completed_papers:
                    paper_table(completed_papers, name)
            if self.result is not None:
                # 已完成运行：下载、预览与审计就近渲染在产出它们的阶段里。
                self.render_result_panel(name)
                return
            if name == PipelineStage.FINAL_RANKED:
                self.render_verification()

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

    def render_result_panel(self, name: PipelineStage) -> None:
        """把下载、预览与审计放回产出它们的阶段。"""
        if self.result is None or self.settings is None:
            return
        if name == PipelineStage.DOWNLOADED:
            stage_panels.render_download_panel(self.result, self.settings)
        elif name == PipelineStage.MARKDOWN_CONVERTED:
            stage_panels.render_conversion_panel(self.result, self.settings)
        elif name == PipelineStage.FINAL_RANKED:
            stage_panels.render_ranking_panel(self.result, self.settings, workbench=self.workbench)
            self.render_verification()
        elif name == PipelineStage.SUMMARIZED:
            stage_panels.render_summaries_panel(self.result, self.settings)
        elif name == PipelineStage.REPORTED:
            stage_panels.render_report_panel(self.result)

    def render_download(self, event: StageEvent) -> None:
        """全文候选池的逐篇下载进度：是否找到、来自哪个渠道、下载是否完成。"""
        children = list(self.children[PipelineStage.DOWNLOADED].values())
        total = event.input_count or len(children)
        done = sum(child.status != ProcessingStatus.RUNNING for child in children)
        rows = [download_row(index, child) for index, child in enumerate(children, 1)]
        found = sum(row["是否找到"].startswith("✅") for row in rows)
        ratio = done / total if total else 0.0
        st.progress(
            min(ratio, 1.0),
            text=f"全文候选池 {total} 篇 · 已处理 {done} 篇 · 已找到公开全文 {found} 篇 · "
            f"待处理 {max(total - done, 0)} 篇",
        )
        st.caption(
            "候选池来自“元数据预排序”的前 N 篇，逐篇按 直链（arXiv/出版商）、Unpaywall、"
            "OpenAlex、落地页 的顺序尝试，任一渠道取到 PDF 即停止。"
        )
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        attempts = fulltext_attempt_rows(children)
        if attempts:
            st.caption("公开全文获取尝试记录（查询成功不等于 PDF 下载成功）")
            st.dataframe(attempts, hide_index=True, width="stretch")
            st.download_button(
                "下载公开全文获取尝试记录 CSV",
                fulltext_attempt_csv(attempts),
                file_name="fulltext_attempts.csv",
                mime="text/csv",
                key=f"{self.key}_fulltext_attempts",
                on_click="ignore",
            )

    def render_conversion(self, event: StageEvent) -> None:
        """PDF 转 Markdown 的逐篇进度：解析器、质量、失败原因。"""
        children = list(self.children[PipelineStage.MARKDOWN_CONVERTED].values())
        legacy = not children
        if legacy:
            children = list(self.children[PipelineStage.DOWNLOADED].values())
        total = event.input_count or len(children)
        done = sum(child.status != ProcessingStatus.RUNNING for child in children)
        rows = [conversion_row(index, child) for index, child in enumerate(children, 1)]
        converted = sum(row["转换状态"].startswith("✅") for row in rows)
        failed = sum(row["转换状态"].startswith("❌") for row in rows)
        skipped = sum(row["转换状态"].startswith("⏭️") for row in rows)
        ratio = done / total if total else 0.0
        st.progress(
            min(ratio, 1.0),
            text=f"待转换 {total} 篇 · 已完成 {done} 篇 · 已生成 Markdown {converted} 篇 · "
            f"失败 {failed} 篇 · 跳过 {skipped} 篇",
        )
        st.caption(
            "解析按固定路数并发（MinerU 为云端排队，单篇上限 600 秒）；解析失败仍保留下载的 PDF，可稍后重试。"
        )
        if legacy:
            st.caption("历史记录：该次运行未分段记录转换阶段，以下信息来自全文处理快照。")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")

    def render_verification(self) -> None:
        """证据确认已并入最终重排，这里以明细表保留可追溯性。"""
        children = list(self.hidden_events.get(PipelineStage.VERIFIED, {}).values())
        children = [child for child in children if child.scope == "paper"]
        if not children:
            return
        with st.expander(f"证据确认明细（已并入本阶段，共 {len(children)} 篇）"):
            st.dataframe(
                [
                    {
                        "文献题目": child.details.get("title"),
                        "证据覆盖度": (
                            (child.details.get("papers") or [{}])[0].get("scores") or {}
                        ).get("evidence_coverage"),
                        "状态": STATUS_LABELS.get(child.status, child.status.value),
                        "说明": child.message,
                    }
                    for child in children
                ],
                hide_index=True,
                width="stretch",
            )
