"""Quality regressions: bilingual recall, topic precision, OA alternatives and bounded reviews."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters.academic_ranker import AcademicFinalRanker, AcademicPreRanker
from paper_agent.adapters.full_text_processor import PdfFullTextProcessor
from paper_agent.adapters.llm_client import OpenAiCompatibleChatClient
from paper_agent.adapters.open_access_resolver import UnpaywallResolver
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.short_review import ShortReviewWriter, review_length
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import (
    FullTextAsset,
    Paper,
    PaperSource,
    PipelineRequest,
    ProcessingStatus,
    ResearchIdea,
)

IDEA = ResearchIdea(text="场景文本检测与识别")


@pytest.mark.asyncio
async def test_chinese_topic_has_bilingual_queries_and_provider_specific_strategies():
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(IDEA, 12)
    for source in (
        PaperSource.ARXIV,
        PaperSource.OPENALEX,
        PaperSource.CROSSREF,
        PaperSource.SEMANTIC_SCHOLAR,
        PaperSource.EUROPE_PMC,
    ):
        queries = [q for q in plan.queries if q.source == source]
        assert {q.language for q in queries} == {"en", "zh"}
        assert any("scene text" in q.query for q in queries)
        assert all(q.strategy for q in queries)
    assert any(
        q.query.startswith("TITLE_ABS:") for q in plan.queries if q.source == PaperSource.EUROPE_PMC
    )
    assert plan.source_limit == 12


@pytest.mark.asyncio
async def test_scene_text_papers_beat_recent_industrial_and_medical_distractors():
    relevant = [
        Paper(title="EAST: An Efficient and Accurate Scene Text Detector", year=2017),
        Paper(
            title="CRAFT: Character Region Awareness for Text Detection",
            abstract="We propose a framework for scene text detection.",
            year=2019,
        ),
        Paper(title="场景文本检测与识别综述", year=2021),
    ]
    noise = [
        Paper(title="基于机器视觉的化工储罐液位仪表故障识别技术", year=2026, citation_count=9999),
        Paper(
            title="[Advances on Postoperative Recurrence Risk Prediction in Lung Cancer]", year=2026
        ),
        Paper(title="Surface Defect Detection for Automotive Wiring", year=2026),
    ]
    for ranker in [AcademicPreRanker(), AcademicFinalRanker()]:
        result = await ranker.rank(IDEA, noise + relevant, 5)
        assert {p.paper_id for p in result} == {p.paper_id for p in relevant}
        assert all(p.scores.final > 0 for p in result)


@pytest.mark.asyncio
async def test_main_task_title_outranks_downstream_application_with_same_abstract():
    idea = ResearchIdea(
        text="场景文本检测与识别",
        search_terms=[
            "scene text detection",
            "scene text recognition",
            "scene text detection and recognition",
        ],
    )
    abstract = "We use scene text detection and scene text recognition for our experiments."
    main = Paper(
        title="Scene Text Detection and Recognition: The Deep Learning Era", abstract=abstract
    )
    downstream = Paper(
        title="Visual place recognition from scene text features",
        abstract=abstract,
        citation_count=10000,
        year=2026,
    )
    result = await AcademicPreRanker().rank(idea, [downstream, main], 2)
    assert result[0].paper_id == main.paper_id


def test_openalex_keeps_all_public_pdf_locations_and_real_landing_page():
    paper = OpenAlexRetriever()._to_paper(
        {
            "id": "https://openalex.org/W123",
            "title": "Scene text detection",
            "primary_location": {"landing_page_url": "https://publisher.example.org/paper"},
            "best_oa_location": {"pdf_url": "https://repo.example.org/a.pdf"},
            "locations": [{"pdf_url": "https://arxiv.org/pdf/1704.03155"}],
        }
    )
    assert str(paper.landing_page_url) == "https://publisher.example.org/paper"
    assert len(paper.fulltext.candidate_urls) == 2


@pytest.mark.asyncio
async def test_failed_primary_pdf_uses_second_copy_and_records_attempts(tmp_path):
    class Resolver:
        async def resolve(self, paper):
            return paper

    class Downloader:
        async def download(self, paper, destination_dir):
            if "broken" in str(paper.fulltext.pdf_url):
                raise PermissionError("HTTP 403")
            path = tmp_path / "paper.pdf"
            path.write_bytes(b"%PDF-1.4 test")
            return path

    class Parser:
        async def parse(self, pdf, destination_dir):
            path = tmp_path / "paper.md"
            path.write_text("# Scene text\nEvidence", encoding="utf-8")
            return path, 0.8

    paper = Paper(
        title="Scene text",
        fulltext=FullTextAsset(
            pdf_url="https://example.org/broken.pdf",
            candidate_urls=["https://repo.example.org/good.pdf"],
        ),
    )
    processor = PdfFullTextProcessor(
        resolver=Resolver(), downloader=Downloader(), parser=Parser(), work_dir=tmp_path
    )
    result = await processor.process(paper)
    assert Path(result.fulltext.markdown_path).is_file()
    assert [a["status"] for a in result.fulltext.attempts] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_public_oa_landing_page_pdf_tag_is_resolved(tmp_path):
    def handle(request):
        return httpx.Response(200, text='<meta name="citation_pdf_url" content="/files/paper.pdf">')

    paper = Paper(
        title="OA paper", open_access_url="https://repo.example.org/article", is_open_access=True
    )
    resolver = UnpaywallResolver(
        email="", discover_locations=True, transport=httpx.MockTransport(handle)
    )
    result = await resolver.resolve(paper)
    assert str(result.fulltext.pdf_url) == "https://repo.example.org/files/paper.pdf"


@pytest.mark.asyncio
async def test_metadata_without_abstract_survives_fulltext_failure():
    class Retriever:
        def __init__(self, name):
            self.source_name = name

        async def search(self, plan):
            return [Paper(title="Scene text detection metadata only", doi="10.1234/scene")]

    class NoFulltext:
        async def process(self, paper):
            raise PermissionError("no public PDF")

    deps = replace(
        build_demo_dependencies(),
        retrievers=[Retriever("arxiv"), Retriever("openalex")],
        fulltext_processor=NoFulltext(),
    )
    result = await ResearchPipeline(deps).run(
        PipelineRequest(idea=IDEA, top_k=1, pre_rank_pool_size=2)
    )
    assert result.status == ProcessingStatus.PARTIAL
    assert result.metadata_papers and result.papers
    assert result.papers[0].doi == "10.1234/scene"
    assert result.papers[0].fulltext.parse_error


@pytest.mark.asyncio
async def test_fulltext_only_policy_still_retains_all_metadata():
    class NoFulltext:
        async def process(self, paper):
            raise PermissionError("no public PDF")

    deps = replace(build_demo_dependencies(), fulltext_processor=NoFulltext())
    result = await ResearchPipeline(deps).run(
        PipelineRequest(idea=IDEA, top_k=1, pre_rank_pool_size=3, require_fulltext=True)
    )
    assert not result.papers
    assert len(result.metadata_papers) == 3


@pytest.mark.asyncio
async def test_review_checks_length_and_retries_short_model_output():
    body = (
        "本次研究围绕场景文本检测与识别展开，已有文献提供了不同的方法描述与证据范围。"
        "当前材料仅支持比较研究问题，无法核实统一数据集上的实验结果。"
    ) * 5
    body = body[:390] + "[1]"

    class Client:
        calls = 0

        async def complete(self, messages, **kwargs):
            self.calls += 1
            return json.dumps({"review": "太短" if self.calls == 1 else body}, ensure_ascii=False)

    client = Client()
    report = await ShortReviewWriter(chat_client=client).write(
        IDEA, [Paper(title="Scene text")], []
    )
    assert client.calls == 2
    assert 300 <= review_length(report.executive_summary) <= 500
    assert "参考文献" in report.markdown


@pytest.mark.asyncio
async def test_offline_review_is_bounded_and_honest_about_missing_evidence():
    report = await ShortReviewWriter().write(IDEA, [Paper(title="Scene text detection")], [])
    assert 300 <= review_length(report.executive_summary) <= 500
    assert "缺少足够材料" in report.executive_summary
    assert "抽取式模板" in report.markdown


@pytest.mark.asyncio
async def test_reasoning_model_temperature_rejection_retries_without_parameter():
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        if "temperature" in payloads[-1]:
            return httpx.Response(400, json={"error": "temperature supports default only"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "valid"}}]})

    client = OpenAiCompatibleChatClient(
        base_url="https://model.example.invalid/v1",
        model="reasoner",
        api_key="fixture",
        transport=httpx.MockTransport(handle),
    )
    assert await client.complete([{"role": "user", "content": "test"}]) == "valid"
    assert len(payloads) == 2
    assert "temperature" not in payloads[1]


@pytest.mark.asyncio
async def test_unknown_chinese_topic_without_translation_is_reported_honestly():
    plan = await HeuristicQueryPlanner(expand_queries=True).plan(
        ResearchIdea(text="珊瑚共生菌的氮循环调控"), 6
    )
    assert plan.warnings
    assert all(q.language == "zh" for q in plan.queries)


@pytest.mark.asyncio
async def test_review_rejects_citations_outside_selected_papers():
    class Client:
        async def complete(self, messages, **kwargs):
            return json.dumps({"review": "仅据提供的材料讨论问题与证据限制。" * 25 + "[99]"})

    report = await ShortReviewWriter(chat_client=Client()).write(
        IDEA, [Paper(title="Scene text")], []
    )
    assert "[99]" not in report.executive_summary
    assert "输出未通过校验" in report.markdown
