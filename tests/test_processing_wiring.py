"""Wiring tests for the group 3 adapters and the production bootstrap branch."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from paper_agent.adapters import build_demo_dependencies, build_processing_components
from paper_agent.adapters.academic_ranker import (
    AcademicEvidenceVerifier,
    AcademicFinalRanker,
    AcademicPreRanker,
)
from paper_agent.adapters.arxiv_retriever import ArxivRetriever
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.adapters.llm_report_writer import INFERENCE_LABEL, LlmReportWriter
from paper_agent.adapters.llm_summarizer import LlmSummarizer
from paper_agent.adapters.markdown_parser import PypdfExtractor
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.paper_merger import PaperMerger
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings
from paper_agent.domain.models import Paper, ResearchIdea

# Empty strings explicitly disable the optional remote backends so the wiring
# tests never depend on keys sitting in the developer machine's environment.
CLEAN_ENV = {
    "OPENAI_API_KEY": "", "PAPER_AGENT_LLM_API_KEY": "",
    "MINERU_API_KEY": "", "UNPAYWALL_EMAIL": "",
}

MARKDOWN_TEXT = """# A Study

## Abstract
We introduce a system for summarizing scientific papers with evidence links.

## Method
We parse PDFs into Markdown and prompt a language model with strict schemas.

## Conclusion
The system produces traceable summaries.

## Limitations
Coverage of scanned PDFs depends on the OCR backend.
"""


def _settings(tmp: str, **overrides) -> Settings:
    return Settings(data_dir=str(Path(tmp) / "data"), **overrides)


class ProcessingWiringTest(unittest.TestCase):
    def test_build_processing_components_returns_group3_types(self):
        with mock.patch.dict(os.environ, CLEAN_ENV), tempfile.TemporaryDirectory() as tmp:
            components = build_processing_components(_settings(tmp))
            data_dir_created = (Path(tmp) / "data").exists()
        self.assertIsInstance(components.fulltext_processor, PdfFullTextProcessor)
        self.assertIsInstance(components.summarizer, LlmSummarizer)
        self.assertIsInstance(components.report_writer, LlmReportWriter)
        # Without OPENAI_API_KEY both LLM consumers must degrade, not crash.
        self.assertIsNone(components.summarizer._client)
        self.assertIsNone(components.report_writer._client)
        # Without MINERU_API_KEY the auto backend falls back to pypdf.
        self.assertIsInstance(
            components.fulltext_processor._parser._extractor, PypdfExtractor
        )
        # Assembling components performs no I/O and creates no work dirs.
        self.assertFalse(data_dir_created)

    def test_unknown_env_is_rejected_with_actionable_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            build_agent(Settings(app_env="bogus"))
        self.assertIn("bogus", str(ctx.exception))

    def test_production_env_wires_group3_into_pipeline(self):
        with mock.patch.dict(os.environ, CLEAN_ENV), tempfile.TemporaryDirectory() as tmp:
            agent = build_agent(_settings(tmp, app_env="production"))
        deps = agent._pipeline.deps
        self.assertIsInstance(deps.fulltext_processor, PdfFullTextProcessor)
        self.assertIsInstance(deps.summarizer, LlmSummarizer)
        self.assertIsInstance(deps.report_writer, LlmReportWriter)

    def test_wired_components_run_offline_summary_and_report_chain(self):
        idea = ResearchIdea(text="evidence-aware scientific paper summarization")
        with mock.patch.dict(os.environ, CLEAN_ENV), tempfile.TemporaryDirectory() as tmp:
            components = build_processing_components(_settings(tmp))
            markdown_path = Path(tmp) / "paper.md"
            markdown_path.write_text(MARKDOWN_TEXT, "utf-8")
            paper = Paper(
                title="A Study",
                abstract="We summarize papers with explicit evidence links.",
            )
            paper.fulltext.markdown_path = str(markdown_path)

            async def run_chain():
                summary = await components.summarizer.summarize(idea, paper)
                report = await components.report_writer.write(
                    idea, [paper], [summary]
                )
                return summary, report

            summary, report = asyncio.run(run_chain())

        self.assertEqual(summary.paper_id, paper.paper_id)
        self.assertTrue(summary.generated_from_fulltext)
        self.assertTrue(summary.evidence)
        self.assertLessEqual(len(summary.evidence[0].quote or ""), 500)
        self.assertIn("抽取式降级", summary.relevance_to_idea)
        self.assertIn("A Study", report.markdown)
        self.assertIn("未来方向", report.markdown)
        self.assertEqual(report.paper_ids, [paper.paper_id])
        self.assertEqual(len(report.evidence), len(summary.evidence))
        self.assertTrue(report.future_directions)
        for direction in report.future_directions:
            self.assertTrue(direction.startswith(INFERENCE_LABEL))


REAL_RETRIEVERS = {
    ArxivRetriever, CrossrefRetriever, OpenAlexRetriever,
    SemanticScholarRetriever, EuropePmcRetriever,
}


@pytest.mark.parametrize("mode", ["demo", "retrieval", "ranking", "production"])
def test_all_modes_preserve_independent_wiring(mode, tmp_path):
    deps = build_agent(Settings(app_env=mode, data_dir=str(tmp_path)))._pipeline.deps
    expected = build_demo_dependencies()
    if mode in {"retrieval", "production"}:
        assert {type(r) for r in deps.retrievers} == REAL_RETRIEVERS
        assert len(deps.retrievers) == 5
        assert isinstance(deps.query_planner, HeuristicQueryPlanner)
        assert isinstance(deps.merger, PaperMerger)
    else:
        assert type(deps.query_planner) is type(expected.query_planner)
        assert type(deps.merger) is type(expected.merger)
        assert [type(r) for r in deps.retrievers] == [type(r) for r in expected.retrievers]
    for field, real_type in {
        "pre_ranker": AcademicPreRanker,
        "verifier": AcademicEvidenceVerifier,
        "final_ranker": AcademicFinalRanker,
    }.items():
        kind = real_type if mode in {"ranking", "production"} else type(getattr(expected, field))
        assert type(getattr(deps, field)) is kind
    for field, real_type in {
        "fulltext_processor": PdfFullTextProcessor,
        "summarizer": LlmSummarizer,
        "report_writer": LlmReportWriter,
    }.items():
        kind = real_type if mode == "production" else type(getattr(expected, field))
        assert type(getattr(deps, field)) is kind


def test_dotenv_loading_and_explicit_settings_priority(tmp_path, monkeypatch):
    from paper_agent.config import load_local_env

    (tmp_path / ".env").write_text(
        "# 仅测试配置，不是真实凭证\nPAPER_AGENT_ENV=production\n"
        "PAPER_AGENT_MAX_CONCURRENCY=2\nPAPER_AGENT_LLM_MODEL='file-model'\n"
        "OPENAI_API_KEY=fixture-not-a-real-key\nPAPER_AGENT_LLM_API_KEY=fixture-only\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_AGENT_LLM_MODEL", "environment-model")
    load_local_env()
    assert Settings.from_env().app_env == "production"
    assert Settings.from_env().llm_model == "environment-model"
    assert os.environ["OPENAI_API_KEY"] == os.environ["PAPER_AGENT_LLM_API_KEY"] == ""
    agent = build_agent()
    assert agent._pipeline.max_concurrency == 2
    assert isinstance(agent._pipeline.deps.fulltext_processor, PdfFullTextProcessor)
    assert agent._pipeline.deps.summarizer._client is None
    assert not agent._pipeline.deps.query_planner.concept_extractor.api_key
    explicit = build_agent(Settings(app_env="demo", max_concurrency=7))
    assert explicit._pipeline.max_concurrency == 7
    assert type(explicit._pipeline.deps.query_planner) is type(
        build_demo_dependencies().query_planner
    )


@pytest.mark.parametrize("planner_key,summary_key", [("", ""), ("fixture", ""), ("", "fixture")])
def test_model_keys_are_independent(planner_key, summary_key, monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_AGENT_LLM_API_KEY", planner_key)
    monkeypatch.setenv("OPENAI_API_KEY", summary_key)
    settings = Settings(
        app_env="production", data_dir=str(tmp_path),
        llm_base_url="https://model.example.invalid/v1", llm_model="explicit-model",
    )
    deps = build_agent(settings)._pipeline.deps
    extractor = deps.query_planner.concept_extractor
    assert extractor.api_key == planner_key
    assert extractor.model == settings.llm_model
    assert extractor.base_url == settings.llm_base_url
    assert (deps.summarizer._client is not None) == bool(summary_key)
    assert deps.summarizer._client is deps.report_writer._client


def test_independent_workbench_loads_configuration_before_settings(tmp_path):
    from paper_agent.bootstrap import build_processing_workbench

    (tmp_path / ".env").write_text(
        "PAPER_AGENT_MARKDOWN_BACKEND=pypdf\nPAPER_AGENT_DATA_DIR=from-file\n", "utf-8"
    )
    # 组件在任务运行时才创建，调用同一工厂验证捕获的是加载后的配置。
    workbench = build_processing_workbench()
    components = workbench._components_factory(None)
    assert components.fulltext_processor._pdf_dir == Path("from-file/fulltext/pdfs")
    assert isinstance(components.fulltext_processor._parser._extractor, PypdfExtractor)
    explicit = build_processing_workbench(Settings(data_dir="explicit"))
    assert explicit._components_factory(None).fulltext_processor._pdf_dir == Path(
        "explicit/fulltext/pdfs"
    )


def test_explicit_empty_unpaywall_setting_does_not_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "fixture@example.invalid")
    components = build_processing_components(Settings(unpaywall_email=""))
    assert components.fulltext_processor._resolver._email == ""
    from_env = build_processing_components(Settings.from_env())
    assert from_env.fulltext_processor._resolver._email == "fixture@example.invalid"


def test_mcp_entrypoint_import_and_contract(monkeypatch):
    import importlib

    monkeypatch.setenv("PAPER_AGENT_ENV", "production")
    server = importlib.import_module("paper_agent.mcp_server")
    contract = server.get_pipeline_contract()
    assert contract.version == "0.2.0"
    assert contract.minimum_sources == 2
    assert len(contract.stages) == 9
    assert {type(r) for r in server.agent._pipeline.deps.retrievers} == REAL_RETRIEVERS


if __name__ == "__main__":
    unittest.main()
