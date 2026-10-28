"""Result panels rendered under the stage that produced them.

下载、预览与审计都挂在对应阶段内，不再集中到最后一段统一总结里，
使每个阶段只展示自己的产出。
"""

from __future__ import annotations

import streamlit as st

from paper_agent.application.processing_benchmark import benchmark_to_markdown
from paper_agent.config import Settings
from paper_agent.domain.models import Paper, PipelineResult
from paper_agent.ui.markdown_exports import (
    MAX_BUNDLE_BYTES,
    embed_paper_images,
    evidence_audit,
    load_paper_markdown,
    load_paper_pdf,
    markdown_filename,
    paper_markdown_files,
    summary_to_markdown,
    zip_markdown_files,
    zip_paper_pdfs,
)
from paper_agent.ui.workbench import render_benchmark

PREVIEW_CHARS = 20_000


def download_pool(result: PipelineResult) -> list[Paper]:
    """导出整个下载候选池，不能只保留进入最终 Top-k 的 PDF 和全文。"""
    return list({p.paper_id: p for p in [*result.metadata_papers, *result.papers]}.values())


def build_markdown_bundle(result: PipelineResult, settings: Settings) -> tuple[dict, bool]:
    """汇总全文、单篇总结、综合报告与性能对比，供整包下载。"""
    pool = download_pool(result)
    by_id = {p.paper_id: p for p in pool}
    downloads: dict = {}
    total_bytes = 0
    for paper in pool:
        for filename, content in paper_markdown_files(paper, settings.data_dir).items():
            name = f"papers/{filename}"
            if name in downloads:
                continue
            total_bytes += len(content.encode("utf-8") if isinstance(content, str) else content)
            if total_bytes > MAX_BUNDLE_BYTES:
                return downloads, True
            downloads[name] = content
    sources = {
        summary.paper_id: (
            load_paper_markdown(by_id[summary.paper_id], settings.data_dir)
            if summary.paper_id in by_id and summary.generated_from_fulltext
            else (by_id[summary.paper_id].abstract if summary.paper_id in by_id else None)
        )
        for summary in result.summaries
    }
    for summary in result.summaries:
        paper = by_id.get(summary.paper_id)
        title = paper.title if paper else "paper"
        filename = markdown_filename(title, summary.paper_id, "summary")
        downloads[f"summaries/{filename}"] = summary_to_markdown(
            summary, paper, source_text=sources[summary.paper_id]
        )
    if result.report and result.report.markdown.strip():
        downloads["research_report.md"] = result.report.markdown
    benchmark = st.session_state.get("processing_benchmark")
    if benchmark and benchmark["run_id"] == str(result.run_id):
        downloads["benchmark.md"] = benchmark_to_markdown(benchmark["report"])
    return downloads, False


def render_download_panel(result: PipelineResult, settings: Settings) -> None:
    """05 论文下载：批量与逐篇原始 PDF，以及完整候选池元数据。"""
    run_id = result.run_id
    pool = download_pool(result)
    if not pool:
        st.info("本次没有可下载的候选论文，请调整研究条件后重试。")
        return
    with st.expander("批量下载原始 PDF", expanded=False):
        # 大批量 PDF 按需打包，常规页面重跑不反复读取整批原文。
        if st.button("准备批量 PDF 下载", key=f"prepare_pdf_{run_id}"):
            st.session_state.pop("pdf_bundle", None)
            try:
                content = zip_paper_pdfs(pool, settings.data_dir)
                if content:
                    st.session_state["pdf_bundle"] = {"run_id": str(run_id), "data": content}
                else:
                    st.info("本批尚无可下载的原始 PDF，请先完成真实论文下载。")
            except ValueError as exc:
                st.warning(str(exc))
        stored_pdf = st.session_state.get("pdf_bundle")
        if stored_pdf and stored_pdf["run_id"] != str(run_id):
            st.session_state.pop("pdf_bundle", None)
            stored_pdf = None
        if stored_pdf:
            st.download_button(
                "一键下载全部论文 PDF（ZIP）",
                stored_pdf["data"],
                file_name=f"papers_{run_id}.zip",
                mime="application/zip",
                key=f"download_pdf_bundle_{run_id}",
            )
        st.caption(f"PDF 打包包含全部 {len(pool)} 篇候选论文（含未进入 Top-k 且已下载的条目）。")

    retained = result.metadata_papers or result.papers
    with st.expander(f"全部文献元数据（{len(retained)} 篇）"):
        st.caption(
            f"保留 {len(retained)} 篇去重后的文献元数据；全文下载或解析失败不会删除这些条目。"
        )
        st.dataframe(
            [
                {
                    "题目": p.title,
                    "作者": ", ".join(a.name for a in p.authors),
                    "年份": p.year,
                    "DOI": p.doi,
                    "arXiv": p.arxiv_id,
                    "来源": ", ".join(dict.fromkeys(r.source.value for r in p.source_records)),
                    "全文状态": "已解析" if p.fulltext.markdown_path else "仅元数据/摘要",
                    "获取失败原因": p.fulltext.parse_error,
                    "摘要": p.abstract,
                }
                for p in retained
            ],
            hide_index=True,
            width="stretch",
        )
        top_ids = {p.paper_id for p in result.papers}
        with st.expander("下载候选文献原始 PDF（包括未进入 Top-k 的论文）"):
            candidate_pdfs = [
                p for p in retained if p.paper_id not in top_ids and p.fulltext.local_pdf_path
            ]
            if not candidate_pdfs:
                st.caption("本批候选论文没有额外的原始 PDF 可下载。")
            for paper in candidate_pdfs:
                pdf = load_paper_pdf(paper, settings.data_dir)
                st.download_button(
                    f"下载 PDF：{paper.title}",
                    pdf or b"",
                    file_name=markdown_filename(paper.title, paper.paper_id, "paper")[:-3] + ".pdf",
                    mime="application/pdf",
                    key=f"metadata_pdf_{run_id}_{paper.paper_id}",
                    disabled=pdf is None,
                )
        st.download_button(
            "下载全部文献元数据 JSON",
            "[" + ",".join(p.model_dump_json() for p in retained) + "]",
            file_name="all_paper_metadata.json",
            mime="application/json",
        )


def _render_missing_markdown(papers: list[Paper]) -> None:
    """没有 Markdown 的论文只列状态与原因，不提供下载按钮。"""
    with st.expander(f"未生成 Markdown 的论文（{len(papers)} 篇）"):
        st.caption(
            "这些论文未取得公开全文或解析失败，因此没有可预览的 Markdown；"
            "失败原因与取得渠道见 05 论文下载。"
        )
        st.dataframe(
            [
                {
                    "文献题目": paper.title,
                    "访问状态": paper.fulltext.access_status.value,
                    "说明": paper.fulltext.parse_error or "未取得可解析的公开全文",
                }
                for paper in papers
            ],
            hide_index=True,
            width="stretch",
        )


def render_conversion_panel(result: PipelineResult, settings: Settings) -> None:
    """06 转换为 Markdown：整包下载与逐篇预览、下载。"""
    run_id = result.run_id
    pool = download_pool(result)
    documents = {p.paper_id: load_paper_markdown(p, settings.data_dir) for p in pool}
    downloads, oversized = build_markdown_bundle(result, settings)
    bundle = None
    if oversized:
        st.warning("全文和图片总大小超过 200 MB，请分篇下载，未生成不完整的批量压缩包。")
    elif downloads:
        try:
            bundle = zip_markdown_files(downloads)
        except ValueError as exc:
            st.warning(str(exc))
    st.download_button(
        "一键下载全部 Markdown（ZIP）",
        bundle or b"",
        file_name=f"research_{run_id}.zip",
        mime="application/zip",
        key=f"bundle_{run_id}",
        disabled=not downloads or bundle is None,
    )
    if not pool:
        st.info("本次没有可转换的论文，请调整研究条件后重试。")
        return
    st.caption("全文 ZIP 包含图片资源，请完整解压后阅读。")
    available = [paper for paper in pool if documents.get(paper.paper_id) is not None]
    st.caption(
        f"可下载全文 {len(available)}/{len(pool)} 篇，"
        f"单篇总结 {len(result.summaries)} 份，综合报告 "
        f"{int(bool(result.report and result.report.markdown.strip()))} 份。"
    )
    if len(available) < len(pool):
        st.info(
            f"其余 {len(pool) - len(available)} 篇尚无可下载全文，压缩包仅包含已存在的文件；"
            "摘要总结不代表论文全文，失败原因可在 05 论文下载中查看。"
        )

    missing = [paper for paper in pool if documents.get(paper.paper_id) is None]
    if missing:
        _render_missing_markdown(missing)
    if not available:
        return
    st.markdown("#### 逐篇预览与下载")
    for paper in available:
        markdown = documents[paper.paper_id]
        with st.expander(f"预览与下载：{paper.title}"):
            st.markdown(embed_paper_images(paper, settings.data_dir, markdown[:PREVIEW_CHARS]))
            if len(markdown) > PREVIEW_CHARS:
                st.caption(f"预览仅显示前 {PREVIEW_CHARS} 字符，下载文件包含完整正文。")
            for warning in paper.fulltext.parse_warnings:
                st.warning(warning)
            st.download_button(
                "下载论文全文 Markdown",
                embed_paper_images(paper, settings.data_dir, markdown),
                file_name=markdown_filename(paper.title, paper.paper_id, "paper"),
                mime="text/markdown",
                key=f"paper_{run_id}_{paper.paper_id}",
                disabled=markdown is None,
            )
            st.download_button(
                "下载全文与图片资源（ZIP）",
                zip_markdown_files(paper_markdown_files(paper, settings.data_dir)),
                file_name=markdown_filename(paper.title, paper.paper_id, "paper")[:-3] + ".zip",
                mime="application/zip",
                key=f"paper_assets_{run_id}_{paper.paper_id}",
            )

    with st.expander("串行 / 多进程解析对照（性能对比）"):
        render_benchmark(result, settings)


def render_ranking_panel(
    result: PipelineResult, settings: Settings, *, workbench: bool = False
) -> None:
    """07 最终重排：Top-k 结果与逐篇明细。"""
    if result.warnings:
        with st.expander("运行提示与降级原因"):
            for warning in result.warnings:
                st.warning(warning)
    if workbench:
        st.caption("此入口仅处理指定论文，未执行相关性排序或代码仓库核验，因此不展示这些分数。")
    documents = {
        paper.paper_id: load_paper_markdown(paper, settings.data_dir) for paper in result.papers
    }
    rows = []
    for rank, paper in enumerate(result.papers, 1):
        row = {
            "序号" if workbench else "排名": rank,
            "题目": paper.title,
            "年份": str(paper.year) if paper.year else "未提供",
            "发表位置": (paper.venue.name if paper.venue else None) or "未提供",
            "DOI": paper.doi or "未提供",
            "arXiv": paper.arxiv_id or "—",
            "开放获取": "是"
            if paper.is_open_access
            else "否"
            if paper.is_open_access is False
            else "未确认",
        }
        if not workbench:
            row.update(
                {
                    "代码状态": paper.code.status.value,
                    "最终分数（满分2）": paper.scores.display_score
                    if paper.scores.display_score is not None
                    else (2 * paper.scores.final if paper.scores.final is not None else None),
                }
            )
        rows.append(row)
    if not rows:
        st.info("本次没有进入最终结果的论文。")
        return
    st.dataframe(rows, hide_index=True, width="stretch")
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
                institutions = ", ".join(paper.institutions) or (
                    "arXiv 未提供机构字段，不代表作者没有单位" if metadata_received else missing
                )
                st.markdown(f"**单位**：{institutions}")
                st.markdown(f"**摘要**：{paper.abstract or missing}")
            with right:
                sources = list(
                    dict.fromkeys(
                        "arXiv" if r.source.value == "arxiv" else r.source.value
                        for r in paper.source_records
                    )
                )
                source = "、".join(sources) or (
                    "arXiv（按输入编号，基本信息未获取）" if paper.arxiv_id else missing
                )
                st.markdown(f"**来源**：{source}")
                landing = paper.landing_page_url or paper.fulltext.landing_page_url
                if landing:
                    st.link_button("查看论文来源", str(landing))
                pdf = load_paper_pdf(paper, settings.data_dir)
                st.download_button(
                    "下载论文原始 PDF",
                    pdf or b"",
                    file_name=markdown_filename(paper.title, paper.paper_id, "paper")[:-3] + ".pdf",
                    mime="application/pdf",
                    key=f"pdf_{result.run_id}_{paper.paper_id}",
                    disabled=pdf is None,
                )
                access = {
                    "open_access": "开放获取",
                    "closed": "非开放获取",
                    "unavailable": "不可用",
                    "unknown": "尚未确认",
                }.get(paper.fulltext.access_status.value, "尚未确认")
                st.markdown(f"**访问状态**：{access}")
                parsed = documents.get(paper.paper_id) is not None
                st.markdown(
                    f"**解析状态**：{'已生成全文 Markdown' if parsed else '尚无可用全文文件'}"
                )
                quality = paper.fulltext.parse_quality
                st.markdown(
                    f"**解析完整度**：{f'{quality:.2f}' if quality is not None else '未评估'}"
                )
                st.caption("完整度是文本覆盖与章节结构的启发式评分，不代表论文质量或总结正确性。")
                if not workbench:
                    scores = paper.scores.model_dump(mode="json", exclude_none=True)
                    if scores:
                        with st.expander("查看排序分项"):
                            st.json(scores)
                    else:
                        st.caption("本次未获得排序分数。")


def render_summaries_panel(result: PipelineResult, settings: Settings) -> None:
    """08 单篇总结：总结正文、原文证据核对与下载。"""
    if not result.summaries:
        st.info("本次运行未生成单篇总结。")
        return
    pool = download_pool(result)
    by_id = {p.paper_id: p for p in pool}
    documents = {p.paper_id: load_paper_markdown(p, settings.data_dir) for p in pool}
    sources = {
        summary.paper_id: (
            documents.get(summary.paper_id)
            if summary.generated_from_fulltext
            else (by_id[summary.paper_id].abstract if summary.paper_id in by_id else None)
        )
        for summary in result.summaries
    }
    for summary in result.summaries:
        paper = by_id.get(summary.paper_id)
        title = paper.title if paper else str(summary.paper_id)
        markdown = summary_to_markdown(summary, paper, source_text=sources[summary.paper_id])
        st.subheader(title)
        st.download_button(
            "下载单篇总结 Markdown",
            markdown,
            file_name=markdown_filename(title, summary.paper_id, "summary"),
            mime="text/markdown",
            key=f"summary_{result.run_id}_{summary.paper_id}",
        )
        summary_column, evidence_column = st.columns([3, 2])
        with summary_column:
            st.markdown(markdown)
        with evidence_column:
            st.markdown("#### 原文证据核对")
            st.caption(
                "只核对引文文字，不代表结论或模型给出的页码已被验证。行号指向下载的 Markdown/摘要。"
            )
            audit = evidence_audit(summary, sources[summary.paper_id])
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


def render_report_panel(result: PipelineResult) -> None:
    """09 综合报告：报告正文、下载与运行审计。"""
    st.caption("1000–1500字中文综述：整合方法路线、已有证据与局限，参考文献另列，不计入正文字数。")
    if result.report and result.report.markdown.strip():
        st.download_button(
            "下载综合报告 Markdown",
            result.report.markdown,
            file_name="research_report.md",
            mime="text/markdown",
            key=f"report_{result.run_id}",
        )
        st.markdown(result.report.markdown)
    else:
        st.info("本次运行未生成综合报告。")
    with st.expander("运行审计（全部阶段事件）"):
        st.dataframe(
            [
                {
                    "阶段": event.stage.value,
                    "范围": event.scope,
                    "数据库": event.source_name,
                    "文献": event.details.get("title"),
                    "状态": event.status.value,
                    "输入": event.input_count,
                    "输出": event.output_count,
                    "耗时（毫秒）": event.duration_ms,
                    "错误或说明": event.message,
                }
                for event in (result.progress_events or result.events)
            ],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "下载完整 JSON",
            result.model_dump_json(indent=2),
            file_name=f"pipeline_result_{result.run_id}.json",
            mime="application/json",
        )
