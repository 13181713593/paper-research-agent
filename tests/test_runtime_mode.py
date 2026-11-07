"""Screenshot regression: normal startup must never silently retrieve fake papers."""

from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_agent import config
from paper_agent.adapters.demo import DemoRetriever
from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings, load_local_env
from paper_agent.domain.models import (
    PipelineContract,
    PipelineRequest,
    ProcessingStatus,
    ResearchIdea,
)

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"
TOPIC = (
    "请你帮我调研在5G/6G网络环境下，云边协同架构中的任务卸载与计算资源调度算法，"
    "并重点分析不同架构下的资源分配策略"
)


def test_unconfigured_startup_wires_real_retrieval_and_processing():
    assert Settings().app_env == Settings.from_env().app_env == "production"
    deps = build_agent()._pipeline.deps
    assert isinstance(deps.query_planner, HeuristicQueryPlanner)
    assert len(deps.retrievers) == 6
    assert not any(isinstance(r, DemoRetriever) for r in deps.retrievers)
    assert isinstance(deps.fulltext_processor, PdfFullTextProcessor)


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", [TOPIC, "5G/6G 云边协同任务卸载与资源分配"])
async def test_default_agent_reaches_existing_bilingual_offloading_fix(topic):
    deps = build_agent()._pipeline.deps
    plan = await deps.query_planner.plan(ResearchIdea(text=topic), 5)
    english = [q.query.lower() for q in plan.queries if q.language == "en"]
    assert any("offloading" in q and "edge" in q for q in english)
    assert any("resource" in q and "edge" in q for q in english)
    assert any("5g" in q or "6g" in q for q in english)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [[], TimeoutError("provider unavailable")])
async def test_failed_or_empty_real_search_never_fabricates_papers(outcome, monkeypatch):
    agent = build_agent()
    for retriever in agent._pipeline.deps.retrievers:
        search = (
            mock.AsyncMock(side_effect=outcome)
            if isinstance(outcome, Exception)
            else mock.AsyncMock(return_value=outcome)
        )
        monkeypatch.setattr(retriever, "search", search)
    demo_search = mock.AsyncMock(side_effect=AssertionError("Must not use fake papers"))
    monkeypatch.setattr(DemoRetriever, "search", demo_search)
    result = await agent.research(
        PipelineRequest(idea=ResearchIdea(text=TOPIC), top_k=1, pre_rank_pool_size=2)
    )
    assert result.status == ProcessingStatus.FAILED
    assert result.papers == result.metadata_papers == []
    assert result.report is None
    assert result.warnings
    demo_search.assert_not_awaited()


def source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    source = root / "src/paper_agent/config.py"
    source.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", "utf-8")
    (root / ".env").write_text("PAPER_AGENT_LLM_MODEL=checkout-model\n", "utf-8")
    monkeypatch.setattr(config, "__file__", str(source))
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    return launch


def test_starting_outside_checkout_still_loads_its_configuration(tmp_path, monkeypatch):
    source_checkout(tmp_path, monkeypatch)
    load_local_env()
    assert Settings.from_env().llm_model == "checkout-model"


def test_current_directory_configuration_precedes_checkout_fallback(tmp_path, monkeypatch):
    launch = source_checkout(tmp_path, monkeypatch)
    (launch / ".env").write_text("PAPER_AGENT_LLM_MODEL=cwd-model\n", "utf-8")
    load_local_env()
    assert Settings.from_env().llm_model == "cwd-model"


def test_checkout_fallback_preserves_explicit_environment(tmp_path, monkeypatch):
    source_checkout(tmp_path, monkeypatch)
    monkeypatch.setenv("PAPER_AGENT_LLM_MODEL", "environment-model")
    load_local_env()
    assert Settings.from_env().llm_model == "environment-model"


def test_explicit_missing_env_file_does_not_load_checkout(tmp_path, monkeypatch):
    source_checkout(tmp_path, monkeypatch)
    load_local_env(tmp_path / "missing.env")
    assert Settings.from_env().llm_model == Settings().llm_model


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["demo", "ranking"])
async def test_explicit_fake_retrieval_labels_papers_report_and_warnings(mode):
    result = await build_agent(Settings(app_env=mode)).research(
        PipelineRequest(idea=ResearchIdea(text=TOPIC), top_k=1, pre_rank_pool_size=2)
    )
    assert result.status != ProcessingStatus.FAILED, result.warnings
    assert result.papers and all(p.title.startswith("[演示数据]") for p in result.papers)
    assert any("虚构示例" in w for w in result.warnings)
    assert "[演示报告]" in result.report.title
    assert "虚构示例" in result.report.markdown


@pytest.mark.parametrize("initial_mode", ["demo", "ranking"])
def test_ui_warns_about_fake_papers_and_rebuilds_cached_agent(initial_mode, monkeypatch):
    agent = mock.Mock()
    agent.contract.return_value = PipelineContract(version="0.2.0", stages=[])
    st.cache_resource.clear()
    try:
        with mock.patch("paper_agent.bootstrap.build_agent", return_value=agent) as build:
            monkeypatch.setenv("PAPER_AGENT_ENV", initial_mode)
            app = AppTest.from_file(str(APP), default_timeout=15).run()
            assert not app.exception
            assert any("论文为虚构示例" in item.value for item in app.warning)
            assert build.call_args.args[0].app_env == initial_mode
            monkeypatch.setenv("PAPER_AGENT_ENV", "production")
            app.run()
            assert not app.exception
            assert not any("论文为虚构示例" in item.value for item in app.warning)
            assert build.call_count == 2
            assert build.call_args.args[0].app_env == "production"
            app.run()
            assert build.call_count == 2
    finally:
        st.cache_resource.clear()
