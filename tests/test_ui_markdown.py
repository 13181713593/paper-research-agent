"""验证 Markdown 导出内容、路径边界及 Streamlit 下载入口。"""

from io import BytesIO
from pathlib import Path
from unittest import mock
from uuid import uuid4
from zipfile import ZipFile

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_agent.domain.models import (
    Author,
    EvidenceRef,
    Paper,
    PaperSource,
    PaperSummary,
    PipelineContract,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    ProcessingStatus,
    ResearchIdea,
    ResearchReport,
    SourceRecord,
)
from paper_agent.ui.markdown_exports import (
    evidence_audit,
    load_paper_markdown,
    markdown_filename,
    summary_to_markdown,
    zip_markdown_files,
)

APP = Path(__file__).resolve().parents[1] / "src/paper_agent/ui/app.py"
FULLTEXT = "# 中文论文\n\n## 方法\n这是真实解析产物的测试正文。\n\n| 方法 | 结果 |\n| --- | --- |\n| A | 1 |\n"


@pytest.fixture
def paper(tmp_path):
    markdown = tmp_path / "fulltext" / "markdown" / "paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(FULLTEXT, encoding="utf-8")
    result = Paper(
        title="测试论文：下载与总结",
        authors=[Author(name="测试作者")],
        year=2026,
        doi="10.1234/test",
        abstract="这是用于验证摘要降级的材料。",
    )
    result.fulltext.markdown_path = str(markdown)
    return result


@pytest.fixture
def summary(paper):
    return PaperSummary(
        paper_id=paper.paper_id,
        core_problem="如何提供可追溯的论文总结？",
        method="解析全文并保留引用。",
        experiments="仅验证下载与导出。",
        contributions=["全文可下载", "总结可下载"],
        limitations=["未测试真实模型输出质量"],
        relevance_to_idea="便于课题相关文献整理。",
        evidence=[
            EvidenceRef(
                paper_id=paper.paper_id,
                section="方法",
                page=2,
                quote="证据第一行\n证据第二行",
                markdown_anchor="method",
            )
        ],
        generated_from_fulltext=True,
    )


@pytest.fixture
def result(paper, summary):
    return PipelineResult(
        status=ProcessingStatus.SUCCEEDED,
        stage=PipelineStage.REPORTED,
        request=PipelineRequest(idea=ResearchIdea(text="验证论文下载和总结生成的完整过程")),
        papers=[paper],
        summaries=[summary],
        report=ResearchReport(
            title="综合论文总结",
            executive_summary="测试概述",
            markdown="# 综合论文总结\n\n## 未来方向\n【系统推断】扩展材料。",
            paper_ids=[paper.paper_id],
        ),
    )


def test_load_fulltext_is_lossless(paper, tmp_path):
    assert load_paper_markdown(paper, tmp_path) == FULLTEXT


@pytest.mark.parametrize("content", [b"", b" \n\t", b"\xff\xfe\x00"])
def test_empty_or_invalid_utf8_is_not_downloadable(paper, tmp_path, content):
    Path(paper.fulltext.markdown_path).write_bytes(content)
    assert load_paper_markdown(paper, tmp_path) is None


def test_missing_file_and_absent_path_are_not_downloadable(paper, tmp_path):
    paper.fulltext.markdown_path = str(tmp_path / "missing.md")
    assert load_paper_markdown(paper, tmp_path) is None
    paper.fulltext.markdown_path = None
    assert load_paper_markdown(paper, tmp_path) is None


def test_outside_directory_and_symlink_are_not_downloadable(paper, tmp_path):
    root = tmp_path / "fulltext"
    outside = tmp_path / "private.md"
    outside.write_text("不得导出的外部文件", encoding="utf-8")
    paper.fulltext.markdown_path = str(outside)
    assert load_paper_markdown(paper, root) is None
    link = root / "linked.md"
    link.symlink_to(outside)
    paper.fulltext.markdown_path = str(link)
    assert load_paper_markdown(paper, root) is None
    paper.fulltext.markdown_path = str(root / ".." / "private.md")
    assert load_paper_markdown(paper, root) is None


def test_non_markdown_and_directory_are_not_downloadable(paper, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("仅测试文件过滤，不是真实凭证", encoding="utf-8")
    paper.fulltext.markdown_path = str(secret)
    assert load_paper_markdown(paper, tmp_path) is None
    directory = tmp_path / "directory.md"
    directory.mkdir()
    paper.fulltext.markdown_path = str(directory)
    assert load_paper_markdown(paper, tmp_path) is None


def test_unreadable_and_large_markdown_are_not_downloadable(paper, tmp_path):
    with mock.patch.object(Path, "open", side_effect=PermissionError):
        assert load_paper_markdown(paper, tmp_path) is None
    with mock.patch("paper_agent.ui.markdown_exports.MAX_MARKDOWN_BYTES", 8):
        assert load_paper_markdown(paper, tmp_path) is None


def test_relative_path_is_supported(paper, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper.fulltext.markdown_path = "fulltext/markdown/paper.md"
    assert load_paper_markdown(paper, "fulltext") == FULLTEXT


def test_summary_contains_sections_provenance_and_evidence(paper, summary):
    markdown = summary_to_markdown(summary, paper)
    for heading in ["核心问题", "方法", "实验", "贡献", "局限", "与研究想法的相关性", "证据引用"]:
        assert f"## {heading}" in markdown
    assert paper.title in markdown
    assert "测试作者" in markdown
    assert paper.doi in markdown
    assert "材料来源：基于全文" in markdown
    assert "> 证据第一行\n> 证据第二行" in markdown
    assert "第 2 页" in markdown
    assert "锚点：method" in markdown


def test_abstract_summary_and_missing_metadata_are_honest(summary):
    summary.generated_from_fulltext = False
    summary.evidence = []
    summary.contributions = []
    markdown = summary_to_markdown(summary)
    assert str(summary.paper_id) in markdown
    assert "仅基于摘要降级生成" in markdown
    assert "未提供可定位的证据引用" in markdown
    assert "材料中未明确说明" in markdown


def test_filenames_are_safe_unique_and_readable():
    title = "../../同名论文\\标题:\n" + "长" * 200
    first = markdown_filename(title, uuid4(), "paper")
    second = markdown_filename(title, uuid4(), "paper")
    assert first != second
    assert "同名论文" in first
    assert all(character not in first for character in ["/", "\\", ":", "\n"])
    assert first.endswith(".md")
    assert len(first.encode("utf-8")) < 255


def test_zip_preserves_utf8_contents(paper, summary):
    files = {
        "papers/全文.md": FULLTEXT,
        "summaries/总结.md": summary_to_markdown(summary, paper),
        "research_report.md": "# 综合报告\n中文内容\n",
    }
    with ZipFile(BytesIO(zip_markdown_files(files))) as archive:
        assert set(archive.namelist()) == set(files)
        for filename, content in files.items():
            assert archive.read(filename).decode("utf-8") == content


@pytest.fixture
def ui(tmp_path, monkeypatch, result):
    monkeypatch.setenv("PAPER_AGENT_ENV", "production")
    monkeypatch.setenv("PAPER_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    agent = mock.Mock()
    agent.contract.return_value = PipelineContract(version="0.2.0", stages=[])
    agent.research = mock.AsyncMock(return_value=result)
    st.cache_resource.clear()
    with (
        mock.patch("paper_agent.bootstrap.build_agent", return_value=agent),
        mock.patch("streamlit.download_button", wraps=st.download_button) as download,
    ):
        app = AppTest.from_file(str(APP), default_timeout=15)
        yield app, agent, download
    st.cache_resource.clear()


def download_calls(download, label):
    return [call for call in download.call_args_list if call.args[0] == label]


def test_ui_generates_three_download_types_without_regeneration(ui, result):
    app, agent, download = ui
    app.run()
    assert not app.exception
    assert not agent.research.called
    assert not download.called
    app.button[0].click().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Top-k 论文",
        "论文 Markdown",
        "单篇总结",
        "综合报告",
        "性能对比",
        "运行审计",
    ]
    paper_call = download_calls(download, "下载论文全文 Markdown")[0]
    assert paper_call.args[1] == FULLTEXT
    assert paper_call.kwargs["mime"] == "text/markdown"
    assert not paper_call.kwargs["disabled"]
    summary_call = download_calls(download, "下载单篇总结 Markdown")[0]
    assert "如何提供可追溯的论文总结" in summary_call.args[1]
    report_call = download_calls(download, "下载综合报告 Markdown")[0]
    assert report_call.args[1] == result.report.markdown
    bundle = download_calls(download, "一键下载全部 Markdown（ZIP）")[0]
    with ZipFile(BytesIO(bundle.args[1])) as archive:
        assert len(archive.namelist()) == 3
        assert archive.read("research_report.md").decode("utf-8") == result.report.markdown
    # 下载触发的界面重跑只读取会话内已有结果，不重复检索或调用 LLM。
    app.run()
    assert not app.exception
    assert agent.research.await_count == 1


@pytest.mark.parametrize("mode,description", [
    ("demo", "全部使用离线演示组件"),
    ("retrieval", "排序、全文处理、总结和报告仍为演示组件"),
    ("ranking", "检索、全文处理和总结仍为演示组件"),
    ("production", "五源真实检索 → 去重 → 第二组排序 → 第三组全文处理"),
])
def test_ui_describes_actual_mode(ui, monkeypatch, mode, description):
    app, _, _ = ui
    monkeypatch.setenv("PAPER_AGENT_ENV", mode)
    app.run()
    assert not app.exception
    assert any(description in item.value for item in app.sidebar.info)
    app.radio[0].set_value("第三组独立演示").run()
    assert not app.exception
    assert any("不执行检索或排序" in item.value for item in app.sidebar.info)


def test_ui_loads_dotenv_before_display_settings(ui, monkeypatch, tmp_path):
    app, _, _ = ui
    monkeypatch.delenv("PAPER_AGENT_ENV")
    (tmp_path / ".env").write_text("PAPER_AGENT_ENV=retrieval\n", "utf-8")
    app.run()
    assert not app.exception
    assert any("retrieval：五源真实检索" in item.value for item in app.sidebar.info)


def test_ui_missing_fulltext_keeps_summary_and_report_downloads(ui, result):
    app, _, download = ui
    result.papers[0].fulltext.markdown_path = None
    result.summaries[0].generated_from_fulltext = False
    app.run().button[0].click().run()
    assert not app.exception
    assert download_calls(download, "下载论文全文 Markdown")[0].kwargs["disabled"]
    summary_call = download_calls(download, "下载单篇总结 Markdown")[0]
    assert "仅基于摘要降级生成" in summary_call.args[1]
    bundle = download_calls(download, "一键下载全部 Markdown（ZIP）")[0]
    with ZipFile(BytesIO(bundle.args[1])) as archive:
        assert len(archive.namelist()) == 2
        assert not any(name.startswith("papers/") for name in archive.namelist())


def test_ui_failed_result_has_no_fake_downloads(ui, result):
    app, _, download = ui
    result.papers = []
    result.summaries = []
    result.report = None
    result.status = ProcessingStatus.FAILED
    result.stage = PipelineStage.FAILED
    app.run().button[0].click().run()
    assert not app.exception
    assert download_calls(download, "一键下载全部 Markdown（ZIP）")[0].kwargs["disabled"]
    assert not download_calls(download, "下载论文全文 Markdown")
    assert not download_calls(download, "下载单篇总结 Markdown")
    assert not download_calls(download, "下载综合报告 Markdown")


def test_ui_same_title_papers_get_unique_download_keys(ui, result):
    app, _, download = ui
    second = result.papers[0].model_copy(deep=True)
    second.paper_id = uuid4()
    second_summary = result.summaries[0].model_copy(deep=True)
    second_summary.paper_id = second.paper_id
    result.papers.append(second)
    result.summaries.append(second_summary)
    app.run().button[0].click().run()
    assert not app.exception
    paper_calls = download_calls(download, "下载论文全文 Markdown")
    assert len(paper_calls) == 2
    assert len({call.kwargs["key"] for call in paper_calls}) == 2
    assert len({call.kwargs["file_name"] for call in paper_calls}) == 2


def test_ui_new_run_replaces_previous_downloads(ui, result):
    app, agent, download = ui
    app.run().button[0].click().run()
    agent.research.return_value = result.model_copy(
        update={
            "run_id": uuid4(),
            "papers": [],
            "summaries": [],
            "report": None,
        }
    )
    download.reset_mock()
    app.button[0].click().run()
    assert not app.exception
    assert agent.research.await_count == 2
    assert not download_calls(download, "下载论文全文 Markdown")
    assert download_calls(download, "一键下载全部 Markdown（ZIP）")[0].kwargs["disabled"]


def test_evidence_audit_locates_real_lines_and_rejects_fabrications(summary):
    source = "# 论文\n\n## 方法\n证据第一行\n证据第二行\n"
    rows = evidence_audit(summary, source)
    assert rows[0]["核对"] == "原文匹配"
    assert rows[0]["起始行"] == 4
    assert rows[0]["结束行"] == 5
    summary.evidence[0].quote = "原文中不存在的结论"
    assert evidence_audit(summary, source)[0]["核对"] == "未定位"
    assert evidence_audit(summary, None)[0]["核对"] == "未定位"


def test_evidence_from_wrong_paper_is_not_verified(summary):
    summary.evidence[0].paper_id = uuid4()
    assert evidence_audit(summary, summary.evidence[0].quote)[0]["核对"] == "未定位"


def test_ui_workbench_does_not_invoke_retrieval(ui, result):
    from paper_agent.application.processing_workbench import PaperProcessingRecord, ProcessingRun

    app, agent, download = ui
    paper = result.papers[0]
    paper.arxiv_id = "1706.03762"
    paper.landing_page_url = "https://arxiv.org/abs/1706.03762"
    paper.fulltext.parse_quality = 0.0
    paper.source_records = [
        SourceRecord(
            source=PaperSource.ARXIV,
            source_id="1706.03762v7",
            source_url=paper.landing_page_url,
            raw_metadata={"metadata_method": "arxiv_atom_api"},
        )
    ]
    run = ProcessingRun(
        result,
        [
            PaperProcessingRecord(
                str(result.papers[0].paper_id),
                result.papers[0].title,
                fulltext_ok=True,
                summary_ok=True,
            )
        ],
        2,
        1.5,
    )
    workbench = mock.Mock()
    workbench.run = mock.AsyncMock(return_value=run)
    with mock.patch("paper_agent.ui.workbench.build_processing_workbench", return_value=workbench):
        app.run().radio[0].set_value("第三组独立演示").run()
        app.button[0].click().run()
    assert not app.exception
    assert not agent.research.called
    assert workbench.run.await_count == 1
    assert app.tabs[0].label == "输入论文（未排序）"
    assert not app.get("json")
    columns = app.tabs[0].dataframe[0].value.columns
    assert "最终分数" not in columns and "代码状态" not in columns
    text = "\n".join(item.value for item in app.tabs[0].markdown)
    assert "测试作者" in text and paper.abstract in text
    assert "**来源**：arXiv" in text
    assert "arXiv 未提供机构字段" in text
    assert "**解析完整度**：0.00" in text
    assert len(app.get("link_button")) == 1
    assert download_calls(download, "下载论文全文 Markdown")
    assert download_calls(download, "下载逐篇处理记录 JSON")


def test_ui_missing_metadata_is_not_claimed_absent(ui, result):
    app, _, _ = ui
    result.papers[0].authors = []
    result.papers[0].abstract = None
    app.run().button[0].click().run()
    assert not app.exception
    text = "\n".join(item.value for item in app.tabs[0].markdown)
    assert "本次未获取" in text
    assert "无摘要" not in text
    assert "未评估" in text
    assert not app.get("json")


def test_ui_full_pipeline_keeps_populated_ranking_fields(ui, result):
    app, _, _ = ui
    result.papers[0].scores.lexical = 0.5
    app.run().button[0].click().run()
    assert not app.exception
    scores = app.get("json")
    assert len(scores) == 1
    assert "lexical" in scores[0].value
    assert "semantic" not in scores[0].value
    assert "最终分数" in app.tabs[0].dataframe[0].value.columns


def test_ui_workbench_bad_input_does_not_start_processing(ui):
    app, agent, _ = ui
    with mock.patch("paper_agent.ui.workbench.build_processing_workbench") as builder:
        app.run().radio[0].set_value("第三组独立演示").run()
        app.text_area(key="workbench_ids").set_value("http://127.0.0.1/private").run()
        app.button[0].click().run()
        assert not builder.called
    assert not agent.research.called
    assert app.error
    assert not app.exception


def test_ui_benchmark_renders_and_joins_download_bundle(ui, result):
    app, agent, download = ui
    second = result.papers[0].model_copy(deep=True)
    second.paper_id = uuid4()
    result.papers.append(second)
    for paper in result.papers:
        paper.fulltext.local_pdf_path = "data/placeholder.pdf"
    measurement = {"round": 1, "workers": 1, "seconds": 2, "successes": 2, "total": 2}
    report = {
        "scope": "本地解析",
        "backend": "pypdf",
        "converter_version": "test",
        "cache_policy": "隔离缓存",
        "timing_policy": "交替顺序",
        "rounds": 2,
        "workers": 2,
        "dataset": [{"paper_id": "a", "sha256": "a"}, {"paper_id": "b", "sha256": "b"}],
        "measurements": [measurement],
        "speedup": 2,
        "comparable": True,
        "serial_seconds": 2,
        "parallel_seconds": 1,
    }
    with mock.patch(
        "paper_agent.ui.workbench.benchmark_pdf_parsing", new=mock.AsyncMock(return_value=report)
    ) as benchmark:
        app.run().button[0].click().run()
        button = next(button for button in app.button if button.label == "运行串行 / 并行对照")
        button.click().run()
    assert not app.exception
    assert benchmark.await_count == 1
    assert agent.research.await_count == 1
    assert download_calls(download, "下载性能对比 Markdown")
    bundle = download_calls(download, "一键下载全部 Markdown（ZIP）")[-1]
    with ZipFile(BytesIO(bundle.args[1])) as archive:
        assert "benchmark.md" in archive.namelist()
