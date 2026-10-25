"""The live page and reruns must expose source results and failed-run history."""

import asyncio
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock

import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.ui.progress import query_result_row

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"


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
    assert len(app.get("download_button")) >= 5
    assert app.metric[1].value == "9"

    app.run()
    assert not app.exception
    assert app.session_state["pipeline_result"].run_id == result.run_id
    assert any(e.value == "02 · 多源数据库检索" for e in app.subheader)
    assert any("文献题目" in table.value.columns for table in app.dataframe)
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
