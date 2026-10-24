"""Offline A-to-pipeline contract checks; other groups use explicit test/demo data."""
import asyncio
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from test_markdown_parser import make_pdf

from paper_agent.adapters import (
    ArxivRetriever,
    ArxivSearchConfig,
    BibTexMapper,
    PaperMerger,
    build_demo_dependencies,
)
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.domain.models import (
    PaperSource,
    PipelineRequest,
    ProcessingStatus,
    ResearchIdea,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('arxiv_fails', [False, True])
async def test_a_components_integrate_with_pipeline_and_partial_failure(arxiv_fails):
    class FixtureClient:
        def results(self, search):
            if arxiv_fails:
                raise TimeoutError('offline integration timeout fixture')
            yield SimpleNamespace(
                entry_id='https://arxiv.org/abs/2401.01234v1',
                title='Offline retrieval integration fixture',
                authors=[SimpleNamespace(name='Jane Doe')],
                published=date(2024, 1, 2), doi='10.1000/integration',
                summary='Explicitly synthetic abstract for an offline contract test.',
            )

    class FixtureSecondSource:
        source_name = 'openalex'

        async def search(self, plan):
            assert any(q.source == PaperSource.OPENALEX for q in plan.queries)
            return [await BibTexMapper().to_paper(
                '@misc{fixture,title={Offline retrieval integration fixture},'
                'author={Jane Doe},doi={https://doi.org/10.1000/INTEGRATION}}',
                source='openalex', supplemental_metadata={
                    'source_id': 'offline-openalex-fixture',
                    'abstract': 'Explicitly synthetic abstract for an offline contract test.',
                    'raw_metadata': {'mode': 'offline_fixture'},
                },
            )]

    deps = replace(
        build_demo_dependencies(),
        retrievers=[ArxivRetriever(client=FixtureClient(), config=ArxivSearchConfig(
            delay_seconds=0, max_retries=0)), FixtureSecondSource()],
        merger=PaperMerger(),
    )
    result = await ResearchPipeline(deps).run(PipelineRequest(
        idea=ResearchIdea(text='Offline retrieval integration contract test'),
        top_k=1, pre_rank_pool_size=2,
    ))
    assert result.status == ProcessingStatus.PARTIAL  # explicit abstract-only fallback
    assert len(result.papers) == len(result.summaries) == 1
    assert result.report is not None
    assert result.summaries[0].generated_from_fulltext is False
    assert result.papers[0].doi == '10.1000/integration'
    deduped = next(event for event in result.events if event.stage.value == 'deduped')
    assert (deduped.input_count, deduped.output_count) == (1 if arxiv_fails else 2, 1)
    expected = {PaperSource.OPENALEX} if arxiv_fails else {PaperSource.ARXIV, PaperSource.OPENALEX}
    assert {r.source for r in result.papers[0].source_records} == expected
    assert any('source arxiv failed' in w for w in result.warnings) == arxiv_fails


@pytest.fixture
def production_fixture(tmp_path, monkeypatch):
    """只替换外部传输；查询、映射、去重、排序、PDF解析与报告均用真实组件。"""
    from paper_agent.bootstrap import build_agent
    from paper_agent.config import Settings

    agent = build_agent(Settings(
        app_env="production", data_dir=str(tmp_path), markdown_backend="pypdf",
        download_retry_limit=0,
    ))
    deps = agent._pipeline.deps
    text = (
        "Scientific evidence summarization links methods and experiments. "
        "The pipeline retrieves public papers and preserves provenance. "
        "Limitations include scanned PDFs and missing metadata."
    )
    row = SimpleNamespace(
        entry_id="https://arxiv.org/abs/2401.01234v1",
        title="Scientific evidence summarization", authors=[SimpleNamespace(name="Jane Doe")],
        published=date(2024, 1, 2), doi="10.1000/integration", summary=text,
        pdf_url="https://arxiv.org/pdf/2401.01234v1",
    )
    client = mock.Mock()
    client.results.side_effect = lambda search: iter([row])
    deps.retrievers[0] = ArxivRetriever(
        client=client, config=ArxivSearchConfig(delay_seconds=0, max_retries=0),
    )
    crossref_fetch = mock.Mock(return_value={"message": {"items": [{
        "title": [row.title], "DOI": "10.1000/INTEGRATION", "abstract": text,
        "author": [{"given": "Jane", "family": "Doe"}],
        "issued": {"date-parts": [[2024, 1, 2]]},
    }]}})
    monkeypatch.setattr(deps.retrievers[1], "_fetch", crossref_fetch)
    for retriever in deps.retrievers[2:]:
        monkeypatch.setattr(retriever, "search", mock.AsyncMock(return_value=[]))
    pdf_response = mock.Mock(return_value=httpx.Response(
        200, content=make_pdf(text), headers={"content-type": "application/pdf"},
    ))
    deps.fulltext_processor._downloader._transport = httpx.MockTransport(pdf_response)
    return agent, crossref_fetch, pdf_response


@pytest.mark.parametrize("source_fails", [False, True])
@pytest.mark.parametrize("fulltext_fails", [False, True])
def test_three_groups_end_to_end(production_fixture, source_fails, fulltext_fails):
    from paper_agent.domain.errors import ProviderUnavailableError

    agent, crossref_fetch, pdf_response = production_fixture
    if source_fails:
        crossref_fetch.side_effect = ProviderUnavailableError("offline source failure")
    if fulltext_fails:
        pdf_response.return_value = httpx.Response(403)
    result = asyncio.run(agent.research(PipelineRequest(
        idea=ResearchIdea(text="scientific evidence summarization"),
        top_k=1, pre_rank_pool_size=2, candidate_limit_per_source=2,
    )))
    expected_status = ProcessingStatus.PARTIAL if source_fails or fulltext_fails else (
        ProcessingStatus.SUCCEEDED
    )
    assert result.status == expected_status, result.warnings
    assert [e.stage.value for e in result.events] == [
        "query_planned", "retrieved", "deduped", "pre_ranked", "fulltext_processed",
        "verified", "final_ranked", "summarized", "reported",
    ]
    assert len(result.papers) == len(result.summaries) == 1
    paper, summary = result.papers[0], result.summaries[0]
    deduped = result.events[2]
    assert (deduped.input_count, deduped.output_count) == (1 if source_fails else 2, 1)
    assert paper.doi == "10.1000/integration"
    sources = {PaperSource.ARXIV} if source_fails else {PaperSource.ARXIV, PaperSource.CROSSREF}
    assert {record.source for record in paper.source_records} == sources
    assert paper.authors and paper.abstract and paper.bibtex
    assert paper.scores.formula_version == "academic-v1-final"
    assert "lexical-fallback" in paper.scores.explanation
    assert f"evidence={'abstract' if fulltext_fails else 'markdown'}" in paper.scores.explanation
    assert summary.generated_from_fulltext is not fulltext_fails
    assert "抽取式降级" in summary.relevance_to_idea
    assert summary.evidence and all(e.paper_id == paper.paper_id for e in summary.evidence)
    assert result.report.paper_ids == [paper.paper_id]
    assert "跨论文对比矩阵" in result.report.markdown
    assert any("source crossref failed" in warning for warning in result.warnings) == source_fails
    assert any("abstract fallback" in warning for warning in result.warnings) == fulltext_fails
    pdf_response.assert_called_once()
    if not fulltext_fails:
        assert Path(paper.fulltext.local_pdf_path).read_bytes().startswith(b"%PDF-")
        assert "Scientific evidence" in Path(paper.fulltext.markdown_path).read_text("utf-8")
        assert paper.fulltext.parser_name == "pypdf"
        assert paper.fulltext.sha256 and paper.fulltext.parse_quality is not None


def test_production_ui_runs_real_offline_chain(production_fixture, monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    agent, _, pdf_response = production_fixture
    monkeypatch.setenv("PAPER_AGENT_ENV", "production")
    monkeypatch.setenv("PAPER_AGENT_DATA_DIR", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"
    st.cache_resource.clear()
    try:
        with mock.patch("paper_agent.bootstrap.build_agent", return_value=agent):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            app.button[0].click().run()
        assert not app.exception
        result = app.session_state["pipeline_result"]
        assert result.status == ProcessingStatus.PARTIAL  # 默认 Top-k=5，夹具只有一篇。
        assert len(result.events) == 9 and result.report
        assert result.summaries[0].generated_from_fulltext
        assert any("可下载全文 1/1" in caption.value for caption in app.caption)
        assert any("Jane Doe" in item.value for item in app.tabs[0].markdown)
        assert len(app.get("download_button")) >= 5
        pdf_response.assert_called_once()
    finally:
        st.cache_resource.clear()


@pytest.mark.skipif(os.getenv("RUN_NETWORK_SMOKE") != "1", reason="显式开启才进行真实五源检索")
def test_production_live_smoke(tmp_path):
    from paper_agent.bootstrap import build_agent
    from paper_agent.config import Settings

    agent = build_agent(Settings(
        app_env="production", data_dir=str(tmp_path), markdown_backend="pypdf",
        max_concurrency=2, download_timeout_seconds=30, download_retry_limit=0,
    ))
    # 全新临时目录确保不是缓存命中。两个模型 Key 均由隔离 fixture 清空。
    deps = agent._pipeline.deps
    assert not deps.query_planner.concept_extractor.api_key
    assert deps.summarizer._client is deps.report_writer._client is None
    result = asyncio.run(agent.research(PipelineRequest(
        idea=ResearchIdea(text="transformer attention neural machine translation"),
        top_k=1, pre_rank_pool_size=5, candidate_limit_per_source=2,
        require_fulltext=True, allow_abstract_fallback=False,
    )))
    print({"阶段": [e.stage.value for e in result.events], "警告": result.warnings})
    assert result.status != ProcessingStatus.FAILED, result.warnings
    assert len(result.events) == 9
    assert len(result.papers) == len(result.summaries) == 1
    paper = result.papers[0]
    assert paper.authors and paper.source_records
    assert Path(paper.fulltext.local_pdf_path).read_bytes().startswith(b"%PDF-")
    markdown = Path(paper.fulltext.markdown_path).read_text("utf-8")
    assert len(markdown) > 1000
    assert result.summaries[0].generated_from_fulltext
    assert paper.scores.formula_version == "academic-v1-final"
    assert result.report and "跨论文对比矩阵" in result.report.markdown
    print({
        "真实论文": paper.title, "来源": [r.source.value for r in paper.source_records],
        "全文字符": len(markdown), "报告字符": len(result.report.markdown),
        "模型调用": False, "结果目录": str(tmp_path),
    })
