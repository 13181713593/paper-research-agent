"""Wiring tests for the group 3 adapters and the production bootstrap branch."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper_agent.adapters import build_processing_components
from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.adapters.llm_report_writer import INFERENCE_LABEL, LlmReportWriter
from paper_agent.adapters.llm_summarizer import LlmSummarizer
from paper_agent.adapters.markdown_parser import PypdfExtractor
from paper_agent.bootstrap import build_agent
from paper_agent.config import Settings
from paper_agent.domain.models import Paper, ResearchIdea

# Empty strings explicitly disable the optional remote backends so the wiring
# tests never depend on keys sitting in the developer machine's environment.
CLEAN_ENV = {"OPENAI_API_KEY": "", "MINERU_API_KEY": "", "UNPAYWALL_EMAIL": ""}

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


if __name__ == "__main__":
    unittest.main()
