import asyncio
import json
import unittest

from paper_agent.adapters.llm_report_writer import INFERENCE_LABEL, LlmReportWriter
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import (
    EvidenceRef,
    Paper,
    PaperSummary,
    ResearchIdea,
)

IDEA = ResearchIdea(text="evidence-aware scientific paper summarization pipelines")


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete(self, messages, *, temperature=0.2):
        item = self.responses.pop(0) if self.responses else ""
        if isinstance(item, Exception):
            raise item
        return item


def _summary(paper: Paper, *, limitations=("Scanned PDFs need OCR.",)) -> PaperSummary:
    return PaperSummary(
        paper_id=paper.paper_id,
        core_problem="Traceability of claims",
        method="PDF to Markdown then prompting",
        experiments="50 papers, human spot checks",
        contributions=["Evidence-linked summaries"],
        limitations=list(limitations),
        relevance_to_idea="Directly matches the idea",
        evidence=[EvidenceRef(paper_id=paper.paper_id, section="abstract", quote="abstract text")],
        generated_from_fulltext=True,
    )


class ReportWriterTest(unittest.TestCase):
    def test_template_report_is_complete_and_labels_inference(self):
        paper_a, paper_b = Paper(title="Paper A"), Paper(title="Paper B")
        summaries = [_summary(paper_a), _summary(paper_b, limitations=())]
        writer = LlmReportWriter(chat_client=None)
        report = asyncio.run(writer.write(IDEA, [paper_a, paper_b], summaries))
        self.assertIn("Paper A", report.markdown)
        self.assertIn("Paper B", report.markdown)
        self.assertIn("未来方向", report.markdown)
        self.assertIn(IDEA.text, report.markdown)
        self.assertEqual(report.paper_ids, [paper_a.paper_id, paper_b.paper_id])
        self.assertEqual(len(report.evidence), 2)
        self.assertTrue(report.future_directions)
        for direction in report.future_directions:
            self.assertTrue(direction.startswith(INFERENCE_LABEL))
        self.assertIn("全文", report.executive_summary)

    def test_template_report_handles_missing_summaries(self):
        paper = Paper(title="Only metadata")
        writer = LlmReportWriter(chat_client=None)
        report = asyncio.run(writer.write(IDEA, [paper], []))
        self.assertIn("未生成总结", report.markdown)
        self.assertTrue(all(d.startswith(INFERENCE_LABEL) for d in report.future_directions))

    def test_llm_report_is_validated_and_directions_labeled(self):
        payload = {
            "title": "LLM 综合报告",
            "executive_summary": "两篇论文的综合概述。",
            "markdown": "# LLM 综合报告\n\n## 未来方向\n- 方向A",
            "future_directions": ["探索更鲁棒的 OCR 流程"],
        }
        client = FakeChatClient([json.dumps(payload)])
        paper = Paper(title="Paper A")
        writer = LlmReportWriter(chat_client=client)
        report = asyncio.run(writer.write(IDEA, [paper], [_summary(paper)]))
        self.assertEqual(report.title, "LLM 综合报告")
        self.assertEqual(report.future_directions, [f"{INFERENCE_LABEL}探索更鲁棒的 OCR 流程"])
        self.assertEqual(report.paper_ids, [paper.paper_id])
        self.assertEqual(len(report.evidence), 1)

    def test_llm_failure_falls_back_to_template(self):
        boom = ProviderUnavailableError("endpoint down")
        client = FakeChatClient([boom, boom])
        paper = Paper(title="Paper A")
        writer = LlmReportWriter(chat_client=client)
        report = asyncio.run(writer.write(IDEA, [paper], [_summary(paper)]))
        self.assertTrue(report.title.startswith("研究报告"))
        self.assertIn("模板合成", report.markdown)


if __name__ == "__main__":
    unittest.main()
