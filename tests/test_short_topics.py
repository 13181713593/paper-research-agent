"""Short research topics are valid; blank edits must not crash or discard a previous run."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import streamlit as st
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import ResearchIdea

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"


@pytest.mark.parametrize("topic", ["场景文本检测与识别", "OCR", "AI", "图"])
def test_short_nonempty_topic_is_valid(topic):
    assert ResearchIdea(text=f"  {topic}  ").text == topic


@pytest.mark.parametrize("topic", ["", "   ", "\n\t"])
def test_blank_topic_is_rejected(topic):
    with pytest.raises(ValidationError):
        ResearchIdea(text=topic)


@pytest.fixture
def ui(monkeypatch):
    agent = PaperResearchAgent(ResearchPipeline(build_demo_dependencies()))
    agent.research = AsyncMock(wraps=agent.research)
    monkeypatch.setattr("paper_agent.bootstrap.build_agent", lambda: agent)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        yield app, agent
    finally:
        st.cache_resource.clear()


@pytest.mark.parametrize("topic", ["场景文本检测与识别", "OCR"])
def test_short_topic_edit_and_immediate_click_complete_without_error(ui, topic):
    app, agent = ui
    # Reproduce a textarea edit and button click arriving in the very same rerun.
    app.text_area[0].set_value(topic)
    app.button[0].click().run()
    assert not app.exception
    agent.research.assert_awaited_once()
    result = app.session_state["pipeline_result"]
    assert result.request.idea.text == topic
    assert result.papers and result.report


def test_blank_edit_and_click_preserve_previous_result(ui):
    app, agent = ui
    app.button[0].click().run()
    previous_id = app.session_state["pipeline_result"].run_id
    app.text_area[0].set_value(" \n\t ")
    app.button[0].click().run()
    assert not app.exception
    assert app.button[0].disabled
    assert any("请输入研究主题" in message.value for message in app.info)
    agent.research.assert_awaited_once()
    assert app.session_state["pipeline_result"].run_id == previous_id


def test_workbench_accepts_short_topic_and_disables_blank_topic(ui):
    app, _ = ui
    app.radio[0].set_value("第三组独立演示").run()
    app.text_area[0].set_value("场景文本检测与识别").run()
    assert not app.exception
    assert not app.button[0].disabled
    app.text_area[0].set_value("  ").run()
    assert not app.exception
    assert app.button[0].disabled
