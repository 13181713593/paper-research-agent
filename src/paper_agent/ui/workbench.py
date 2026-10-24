"""第三组独立演示及实测面板，界面仅调用应用用例。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import streamlit as st

from paper_agent.application.processing_benchmark import (
    benchmark_pdf_parsing,
    benchmark_to_markdown,
)
from paper_agent.application.processing_workbench import papers_from_arxiv_ids
from paper_agent.bootstrap import build_processing_workbench
from paper_agent.config import Settings
from paper_agent.domain.models import PipelineResult, ResearchIdea


def render_workbench(settings: Settings, idea: str) -> None:
    with st.container(border=True):
        st.subheader("第三组 · 公开论文处理工作台")
        st.caption(
            "指定公开论文 → 获取基本信息 → 下载全文 → 多进程解析 → 并发总结 → 综合报告。"
            "这是第三组独立演示，不代表已完成多源检索、MCP 调用或 Top-k 排序。"
        )
        ids = st.text_area(
            "arXiv 论文编号（每行一个）",
            "1706.03762\n1409.0473",
            key="workbench_ids",
            help="预填两篇真实公开论文，可换成你的论文编号；最多 20 篇。",
        )
        concurrency = st.number_input("下载与总结并发上限", 1, 8, 4, key="workbench_concurrency")
        st.caption(
            "进度日志按完成先后显示；已处理数量包含成功、失败和跳过，"
            "“第几篇”是输入序号。最终论文和总结仍按输入顺序展示。"
        )
        st.caption(
            "该入口始终使用真实处理组件。配置总结 API 后会发送论文材料；未配置则使用抽取式总结。"
        )
        if st.button("处理论文并生成报告", type="primary", disabled=len(idea.strip()) < 10):
            try:
                papers = papers_from_arxiv_ids(ids)
                with st.status("第三组正在处理真实论文……", expanded=True) as status:
                    run = asyncio.run(
                        build_processing_workbench(settings).run(
                            ResearchIdea(text=idea),
                            papers,
                            concurrency=concurrency,
                            progress=st.write,
                        )
                    )
                    status.update(
                        label=f"处理结束：{run.result.status.value}",
                        state="error" if run.result.status.value == "failed" else "complete",
                    )
                st.session_state["pipeline_result"] = run.result
                st.session_state["processing_run"] = run
                st.session_state.pop("processing_benchmark", None)
            except Exception as exc:  # UI 边界保留旧结果并明确告知本次未完成。 # noqa: BLE001
                st.error(
                    f"本次处理未完成：{type(exc).__name__}。请检查编号、网络及解析依赖后重试。"
                )
                st.caption("下方若有结果，为上一次已完成的任务。")


def render_processing_metrics(result: PipelineResult) -> None:
    run = st.session_state.get("processing_run")
    if run is None or run.result.run_id != result.run_id:
        return
    st.info("以下为第三组指定论文的处理结果，顺序与输入一致，未进行检索或相关性排名。")
    first, second, third = st.columns(3)
    first.metric(
        "全文处理成功率", f"{sum(r.fulltext_ok for r in run.records) / len(run.records):.0%}"
    )
    second.metric(
        "单篇总结成功率", f"{sum(r.summary_ok for r in run.records) / len(run.records):.0%}"
    )
    third.metric("批次墙钟耗时", f"{run.elapsed_seconds:.2f} 秒")
    with st.expander("逐篇处理耗时与失败原因"):
        st.dataframe(
            [
                {
                    "论文": row.title,
                    "全文秒": round(row.fulltext_seconds, 3),
                    "总结秒": round(row.summary_seconds, 3),
                    "全文成功": row.fulltext_ok,
                    "总结成功": row.summary_ok,
                    "错误": "；".join(row.errors),
                }
                for row in run.records
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "下载逐篇处理记录 JSON",
            json.dumps(
                {
                    "run_id": str(result.run_id),
                    "concurrency": run.concurrency,
                    "elapsed_seconds": run.elapsed_seconds,
                    "records": [asdict(row) for row in run.records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="processing_metrics.json",
            mime="application/json",
        )


def render_benchmark(result: PipelineResult, settings: Settings) -> None:
    st.subheader("优化亮点 · 串行 / 多进程对照")
    st.caption(
        "用本批已下载的 PDF 实测解析阶段：相同样本、相同 pypdf 后端、"
        "每轮全新解析目录、交替顺序各跑两次，计入进程启动开销。"
        "不重新下载、不调用 LLM 或 MinerU。"
    )
    count = sum(bool(paper.fulltext.local_pdf_path) for paper in result.papers)
    workers = st.number_input("对照实验并行进程数", 2, 8, 4, key="benchmark_workers")
    if count < 2:
        st.info("至少成功下载两篇不同论文后，才能运行本地性能对比。")
    if st.button("运行串行 / 并行对照", disabled=count < 2):
        try:
            with st.spinner("正在进行实际对照实验，请勿重复点击……"):
                report = asyncio.run(
                    benchmark_pdf_parsing(
                        result.papers,
                        settings.data_dir,
                        workers=workers,
                    )
                )
            st.session_state["processing_benchmark"] = {
                "run_id": str(result.run_id),
                "report": report,
            }
            st.rerun()
        except Exception as exc:  # 失败不显示未经完成的加速数字。 # noqa: BLE001
            st.error(f"对照实验未完成：{type(exc).__name__}。请确认本地 PDF 可用且内容不重复。")
    stored = st.session_state.get("processing_benchmark")
    if not stored or stored["run_id"] != str(result.run_id):
        return
    report = stored["report"]
    first, second, third = st.columns(3)
    first.metric("串行中位耗时", f"{report['serial_seconds']:.3f} 秒")
    second.metric("并行中位耗时", f"{report['parallel_seconds']:.3f} 秒")
    speedup = report["speedup"]
    third.metric("实测加速比", f"{speedup:.2f}×" if speedup is not None else "不可比较")
    if speedup is None:
        st.warning("存在失败或输入/输出不一致，不据此计算加速比。")
    elif speedup <= 1:
        st.info("本次未加速。小批量论文的进程启动开销可能超过收益，按实测结果展示。")
    st.bar_chart(
        {
            "模式": ["串行", "并行"],
            "耗时（秒）": [
                report["serial_seconds"],
                report["parallel_seconds"],
            ],
        },
        x="模式",
        y="耗时（秒）",
    )
    st.dataframe(
        [
            {key: value for key, value in row.items() if key != "output_sha256"}
            for row in report["measurements"]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "下载性能对比 Markdown",
        benchmark_to_markdown(report),
        file_name="benchmark.md",
        mime="text/markdown",
    )
    st.download_button(
        "下载性能原始数据 JSON",
        json.dumps(report, ensure_ascii=False, indent=2),
        file_name="benchmark.json",
        mime="application/json",
    )
