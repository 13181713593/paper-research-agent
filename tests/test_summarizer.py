import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from paper_agent.adapters.llm_summarizer import NOT_STATED, LlmSummarizer
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import Paper, ResearchIdea

IDEA = ResearchIdea(text="evidence-aware scientific paper summarization pipelines")

MARKDOWN_TEXT = """# A Study

## Abstract
We introduce a system for summarizing scientific papers. It links claims to evidence.

## Method
We parse PDFs into Markdown and prompt a language model with strict schemas.

## Experiments
We evaluate on 50 papers with human spot checks.

## Conclusion
The system produces traceable summaries. It reduces hallucination risk.

## Limitations
Coverage of scanned PDFs depends on the OCR backend.
"""

GOOD_PAYLOAD = {
    "core_problem": "Scientific summaries lack evidence traceability.",
    "method": "PDF to Markdown plus schema-constrained prompting.",
    "experiments": "50 papers with human spot checks.",
    "contributions": ["Evidence-linked summaries"],
    "limitations": ["OCR quality is a bottleneck"],
    "relevance_to_idea": "Directly matches the idea.",
    "evidence": [{"section": "Abstract", "page": 1, "quote": "x" * 600}],
}


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, messages, *, temperature=0.2):
        self.requests.append(messages)
        item = self.responses.pop(0) if self.responses else ""
        if isinstance(item, Exception):
            raise item
        return item


def _paper_with_markdown(tmp: str) -> Paper:
    markdown_path = Path(tmp) / "paper.md"
    markdown_path.write_text(MARKDOWN_TEXT, "utf-8")
    paper = Paper(title="A Study")
    paper.fulltext.markdown_path = str(markdown_path)
    return paper


class SummarizerTest(unittest.TestCase):
    def test_llm_summary_is_validated_and_evidence_linked(self):
        client = FakeChatClient([json.dumps(GOOD_PAYLOAD)])
        summarizer = LlmSummarizer(chat_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            paper = _paper_with_markdown(tmp)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertEqual(summary.paper_id, paper.paper_id)
        self.assertTrue(summary.generated_from_fulltext)
        self.assertEqual(summary.core_problem, GOOD_PAYLOAD["core_problem"])
        self.assertEqual(len(summary.evidence), 1)
        self.assertEqual(summary.evidence[0].paper_id, paper.paper_id)
        self.assertEqual(summary.evidence[0].section, "Abstract")
        self.assertLessEqual(len(summary.evidence[0].quote or ""), 500)

    def test_llm_bad_output_falls_back_to_extractive(self):
        client = FakeChatClient(["not json at all", "still not json"])
        summarizer = LlmSummarizer(chat_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            paper = _paper_with_markdown(tmp)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertTrue(summary.generated_from_fulltext)
        self.assertIn("We introduce a system", summary.core_problem)
        self.assertIn("抽取式降级", summary.relevance_to_idea)
        self.assertTrue(summary.evidence)
        self.assertEqual(len(client.requests), 2)

    def test_llm_retry_then_success(self):
        client = FakeChatClient(["broken", json.dumps(GOOD_PAYLOAD)])
        summarizer = LlmSummarizer(chat_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            paper = _paper_with_markdown(tmp)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertEqual(summary.core_problem, GOOD_PAYLOAD["core_problem"])

    def test_provider_error_degrades_without_raising(self):
        boom = ProviderUnavailableError("endpoint down")
        client = FakeChatClient([boom, boom])
        summarizer = LlmSummarizer(chat_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            paper = _paper_with_markdown(tmp)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertTrue(summary.generated_from_fulltext)

    def test_abstract_only_sets_flag_false_and_never_invents_experiments(self):
        summarizer = LlmSummarizer(chat_client=None)
        paper = Paper(
            title="Abstract-only paper",
            abstract="We present a dataset for testing. It contains 10 tasks.",
        )
        summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertFalse(summary.generated_from_fulltext)
        self.assertEqual(summary.experiments, NOT_STATED)
        self.assertEqual(summary.core_problem, "We present a dataset for testing. It contains 10 tasks.")
        self.assertEqual(summary.evidence[0].section, "abstract")

    def test_no_evidence_at_all_produces_placeholder_summary(self):
        summarizer = LlmSummarizer(chat_client=None)
        paper = Paper(title="Nothing available")
        summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertFalse(summary.generated_from_fulltext)
        self.assertEqual(summary.evidence, [])
        self.assertIn("无法获取", summary.core_problem)

    def test_extractive_limitations_and_contributions_are_quoted(self):
        summarizer = LlmSummarizer(chat_client=None)
        with tempfile.TemporaryDirectory() as tmp:
            paper = _paper_with_markdown(tmp)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertTrue(summary.contributions)
        self.assertIn("The system produces traceable summaries", summary.contributions[0])
        self.assertTrue(summary.limitations)
        self.assertIn("OCR backend", summary.limitations[0])

    def test_long_fulltext_keeps_conclusion_despite_huge_references(self):
        # Regression from a real arXiv run: the bibliography filled the tail of
        # the truncated text, so conclusion/method evidence disappeared.
        filler = "Body sentence about attention models. " * 900
        references = "\n".join(f"[{i}] Author. Some cited work. 2017." for i in range(2000))
        markdown = (
            "# Long Paper\n\n## Abstract\nThe abstract sentence.\n\n"
            f"## 1 Introduction\n{filler}\n\n## 3 Model Architecture\nEncoder and decoder stacks.\n\n"
            "## 7 Conclusion\nWe presented the Transformer.\n\n"
            f"## References\n{references}\n"
        )
        summarizer = LlmSummarizer(chat_client=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.md"
            path.write_text(markdown, "utf-8")
            paper = Paper(title="Long Paper")
            paper.fulltext.markdown_path = str(path)
            summary = asyncio.run(summarizer.summarize(IDEA, paper))
        self.assertFalse(summary.core_problem.startswith("#"))
        self.assertEqual(summary.core_problem, "The abstract sentence.")
        self.assertIn("Encoder and decoder stacks.", summary.method)
        self.assertIn("We presented the Transformer.", summary.contributions[0])
        sections = {ref.section for ref in summary.evidence}
        self.assertEqual(sections, {"abstract", "method", "conclusion"})


if __name__ == "__main__":
    unittest.main()
