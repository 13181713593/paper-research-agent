"""The live page and reruns must expose source results and failed-run history."""

import asyncio
import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock
from uuid import uuid4

import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import PipelineStage, ProcessingStatus, StageEvent
from paper_agent.ui.progress import (
    PipelineProgress,
    conversion_row,
    download_row,
    fulltext_attempt_csv,
    fulltext_attempt_rows,
    query_result_row,
)

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"


def test_attempt_csv_exports_all_candidates_and_escapes_formula_titles():
    children = [
        StageEvent(
            stage=PipelineStage.FULLTEXT_PROCESSED,
            scope="paper",
            status=ProcessingStatus.PARTIAL,
            started_at=datetime.now(UTC),
            details={
                "title": "=危险外部标题",
                "papers": [
                    {
                        "doi": "10.1000/a",
                        "fulltext": {
                            "attempts": [
                                {"stage": "unpaywall", "status": "skipped", "reason": "缺少 DOI"},
                                {
                                    "stage": "landing_page",
                                    "status": "succeeded",
                                    "candidate_count": "0",
                                },
                            ]
                        },
                    }
                ],
            },
        )
    ]
    rows = fulltext_attempt_rows(children)
    assert len(rows) == 2
    payload = fulltext_attempt_csv(rows)
    assert payload.startswith(b"\xef\xbb\xbf")
    restored = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert restored[0]["文献"] == "'=危险外部标题"
    assert restored[0]["DOI"] == "10.1000/a"
    assert restored[1]["candidate_count"] == "0"
    assert restored[0]["序号"] == "1"


def test_attempt_csv_hides_signed_url_parameters_without_changing_input():
    rows = [
        {
            "stage": "download",
            "url": "https://cdn.example.org/p.pdf?Signature=private-signature&doi=10.1000/a",
        }
    ]
    payload = fulltext_attempt_csv(rows).decode("utf-8-sig")
    assert "private-signature" not in payload
    assert "doi=10.1000/a" in payload
    assert "private-signature" in rows[0]["url"]


def test_query_table_distinguishes_empty_matches_and_actual_request_errors():
    normal_empty = {"query": "中文", "status": "succeeded", "count": 0}
    historical_empty = {
        "query": "中文",
        "status": "failed",
        "count": 0,
        "error": "arXiv query returned no results: 中文",
    }
    timed_out = {
        "query": "English",
        "status": "failed",
        "count": 0,
        "error": "arXiv request timed out",
    }
    assert query_result_row(normal_empty)["状态"] == "未匹配"
    assert query_result_row(historical_empty)["状态"] == "未匹配"
    assert query_result_row(timed_out)["状态"] == "请求失败"
    assert query_result_row(timed_out)["说明"] == timed_out["error"]


def test_page_displays_all_stage_results_and_retains_them_on_rerun(monkeypatch):
    agent = PaperResearchAgent(ResearchPipeline(build_demo_dependencies()))
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["pipeline_result"]
    assert any(e.scope == "source" for e in result.progress_events)
    assert not any(e.label.startswith("2. 多源数据库检索") for e in app.expander)
    stage_buttons = [b for b in app.button if "_stage_" in (b.key or "")]
    assert len(stage_buttons) == 9
    assert all(not b.disabled for b in stage_buttons)
    assert any(e.value == "09 · 综合报告" for e in app.subheader)
    stage_buttons[1].click().run()
    assert not app.exception
    assert any(e.value == "02 · 多源数据库检索" for e in app.subheader)
    markdown = "\n".join(e.value for e in app.markdown)
    assert "arXiv" in markdown and "OpenAlex" in markdown
    assert any("文献题目" in table.value.columns for table in app.dataframe)
    stage_buttons = [b for b in app.button if "_stage_" in (b.key or "")]
    stage_buttons[5].click().run()
    assert not app.exception
    # 06 内联整包下载；逐篇下载取决于本地是否已生成 Markdown 文件。
    assert len(app.get("download_button")) >= 1
    assert any("未生成 Markdown 的论文" in e.label for e in app.expander)
    assert app.metric[1].value == "9"

    app.run()
    assert not app.exception
    assert app.session_state["pipeline_result"].run_id == result.run_id
    assert any(e.value == "06 · 转换为 Markdown" for e in app.subheader)
    assert len(app.get("download_button")) >= 1
    st.cache_resource.clear()


def test_failed_page_keeps_successful_source_papers_and_shows_errors(monkeypatch):
    class BrokenRanker:
        async def rank(self, idea, papers, limit):
            raise ValueError("ranking failed in UI fixture")

    deps = build_demo_dependencies()
    deps.pre_ranker = BrokenRanker()
    monkeypatch.setattr(
        "paper_agent.bootstrap.build_agent", lambda: PaperResearchAgent(ResearchPipeline(deps))
    )
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["pipeline_result"].status.value == "failed"
    assert any("ranking failed in UI fixture" in e.value for e in app.error)
    stage_buttons = [b for b in app.button if "_stage_" in (b.key or "")]
    assert sum(b.disabled for b in stage_buttons) == 5
    stage_buttons[1].click().run()
    assert any("文献题目" in table.value.columns for table in app.dataframe)
    stage_buttons = [b for b in app.button if "_stage_" in (b.key or "")]
    stage_buttons[3].click().run()
    app.run()
    assert not app.exception
    assert any("ranking failed in UI fixture" in e.value for e in app.error)
    st.cache_resource.clear()


def test_idle_navigator_shows_nine_pending_stages(monkeypatch):
    agent = PaperResearchAgent(ResearchPipeline(build_demo_dependencies()))
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    buttons = [b for b in app.button if "_stage_" in (b.key or "")]
    assert len(buttons) == 9
    assert all(b.disabled and "待执行" in b.label for b in buttons)
    assert not any(" · " in h.value for h in app.subheader)
    st.cache_resource.clear()


def test_navigating_completed_stage_while_running_keeps_same_worker(monkeypatch):
    entered, release = Event(), Event()
    deps = build_demo_dependencies()
    original = deps.pre_ranker

    class PausedRanker:
        async def rank(self, idea, papers, limit):
            entered.set()
            await asyncio.to_thread(release.wait, 15)
            return await original.rank(idea, papers, limit)

    deps.pre_ranker = PausedRanker()
    agent = PaperResearchAgent(ResearchPipeline(deps))
    agent.research = AsyncMock(wraps=agent.research)
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[0].click().run()
        assert entered.wait(5)
        job = app.session_state["research_job"]
        assert app.button[0].disabled
        buttons = [b for b in app.button if "_stage_" in (b.key or "")]
        assert "进行中" in buttons[3].label
        assert buttons[4].disabled
        buttons[0].click().run()
        assert not app.exception
        assert app.session_state["research_job"] is job
        assert any(h.value == "01 · 查询规划" for h in app.subheader)
        assert agent.research.await_count == 1
        assert not job.snapshot()[3]
        release.set()
        job._thread.join(timeout=5)
        app.run()
        assert not app.exception
        assert "pipeline_result" in app.session_state
        assert any(h.value == "01 · 查询规划" for h in app.subheader)
        next(b for b in app.button if b.label == "查看最后阶段").click().run()
        assert any(h.value == "09 · 综合报告" for h in app.subheader)
        assert agent.research.await_count == 1
    finally:
        release.set()
        st.cache_resource.clear()


def _paper_event(stage, *, status, fulltext, duration_ms=1500, message=None):
    return StageEvent(
        stage=stage,
        scope="paper",
        paper_id=uuid4(),
        status=status,
        started_at=datetime.now(UTC),
        duration_ms=duration_ms,
        message=message,
        details={"title": "示例文献", "papers": [{"fulltext": fulltext}]},
    )


def test_download_row_reports_channel_host_and_local_file():
    event = _paper_event(
        PipelineStage.DOWNLOADED,
        status=ProcessingStatus.SUCCEEDED,
        fulltext={
            "access_status": "open_access",
            "local_pdf_path": "/tmp/paper.pdf",
            "attempts": [
                {"stage": "unpaywall", "status": "succeeded"},
                {
                    "stage": "download",
                    "status": "succeeded",
                    "channel": "openalex",
                    "url": "https://europepmc.org/api/paper.pdf",
                },
            ],
        },
    )
    row = download_row(1, event)
    assert row["是否找到"].startswith("✅")
    assert row["取得来源"] == "OpenAlex（europepmc.org）"
    assert row["本地文件"] == "已下载 PDF"
    assert row["耗时（秒）"] == 1.5


def test_download_row_marks_missing_fulltext_and_cache_hits():
    missing = download_row(
        1,
        _paper_event(
            PipelineStage.DOWNLOADED,
            status=ProcessingStatus.PARTIAL,
            fulltext={"access_status": "unavailable", "attempts": []},
            message="未取得公开全文：NoOpenAccessCopyError",
        ),
    )
    assert missing["是否找到"].startswith("❌")
    assert missing["取得来源"] == "—"
    assert missing["本地文件"] == "—"

    cached = download_row(
        2,
        _paper_event(
            PipelineStage.DOWNLOADED,
            status=ProcessingStatus.SUCCEEDED,
            fulltext={"markdown_path": "/tmp/a.md", "attempts": []},
        ),
    )
    assert cached["是否找到"].startswith("✅")
    assert cached["取得来源"] == "本地缓存"
    assert cached["本地文件"] == "复用已解析 Markdown"


def test_conversion_row_distinguishes_converted_skipped_and_failed():
    converted = conversion_row(
        1,
        _paper_event(
            PipelineStage.MARKDOWN_CONVERTED,
            status=ProcessingStatus.SUCCEEDED,
            fulltext={"markdown_path": "/tmp/a.md", "parser_name": "mineru", "parse_quality": 0.9},
        ),
    )
    assert converted["转换状态"].startswith("✅")
    assert converted["解析器"] == "mineru"

    skipped = conversion_row(
        2,
        _paper_event(
            PipelineStage.MARKDOWN_CONVERTED,
            status=ProcessingStatus.SKIPPED,
            fulltext={},
            message="未取得公开全文，未进入 Markdown 转换。",
        ),
    )
    assert skipped["转换状态"].startswith("⏭️")

    failed = conversion_row(
        3,
        _paper_event(
            PipelineStage.MARKDOWN_CONVERTED,
            status=ProcessingStatus.PARTIAL,
            fulltext={"parse_error": "timeout"},
            message="PDF 解析超过600秒预算",
        ),
    )
    assert failed["转换状态"].startswith("❌")


def test_progress_aliases_legacy_fulltext_and_hides_verification():
    progress = PipelineProgress(None)
    assert len(progress.stages) == 9
    assert progress.stages[4] == PipelineStage.DOWNLOADED
    assert progress.stages[5] == PipelineStage.MARKDOWN_CONVERTED

    legacy = _paper_event(
        PipelineStage.FULLTEXT_PROCESSED,
        status=ProcessingStatus.SUCCEEDED,
        fulltext={"markdown_path": "/tmp/a.md"},
    )
    progress.update(legacy)
    assert len(progress.children[PipelineStage.DOWNLOADED]) == 1
    assert PipelineStage.FULLTEXT_PROCESSED not in progress.latest

    verified = _paper_event(PipelineStage.VERIFIED, status=ProcessingStatus.SUCCEEDED, fulltext={})
    progress.update(verified)
    assert len(progress.hidden_events[PipelineStage.VERIFIED]) == 1
    assert PipelineStage.VERIFIED not in progress.latest


def test_download_and_conversion_stages_show_pool_and_per_paper_progress(monkeypatch):
    agent = PaperResearchAgent(ResearchPipeline(build_demo_dependencies()))
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    stage_buttons = [b for b in app.button if "_stage_" in (b.key or "")]

    stage_buttons[4].click().run()
    assert not app.exception
    assert any(e.value == "05 · 论文下载" for e in app.subheader)
    progress_text = "\n".join(getattr(p.proto, "text", "") or "" for p in app.get("progress"))
    assert "全文候选池" in progress_text
    assert "已找到公开全文" in progress_text
    captions = "\n".join(e.value for e in app.caption)
    assert "候选池来自" in captions
    download_tables = [table.value for table in app.dataframe if "取得来源" in table.value.columns]
    assert download_tables
    assert download_tables[0]["取得来源"].str.contains("直链").any()

    stage_buttons[5].click().run()
    assert not app.exception
    assert any(e.value == "06 · 转换为 Markdown" for e in app.subheader)
    progress_text = "\n".join(getattr(p.proto, "text", "") or "" for p in app.get("progress"))
    assert "待转换" in progress_text and "已生成 Markdown" in progress_text
    captions = "\n".join(e.value for e in app.caption)
    assert "解析按固定路数并发" in captions
    assert any(
        {"转换状态", "解析器", "Markdown 文件"} <= set(table.value.columns)
        for table in app.dataframe
    )
