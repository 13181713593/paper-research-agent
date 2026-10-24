"""为界面提供 Markdown 文件内容与内存打包，不重新调用生成接口。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

from paper_agent.domain.models import Paper, PaperSummary

MAX_MARKDOWN_BYTES = 20_000_000


def markdown_filename(title: str, identifier: UUID, kind: str) -> str:
    """保留可读标题，用论文标识区分同名文件并移除路径字符。"""
    name = re.sub(r"[^\w .-]", "_", title).strip(" ._")[:50] or "paper"
    return f"{kind}_{name}_{identifier}.md"


def load_paper_markdown(paper: Paper, data_dir: str | Path) -> str | None:
    """仅导出数据目录内存在的 Markdown 产物，缺失或不可读时返回空。"""
    if not paper.fulltext.markdown_path:
        return None
    try:
        root = Path(data_dir).resolve()
        path = Path(paper.fulltext.markdown_path).resolve(strict=True)
        # 拒绝越界路径与指向数据目录外的软链接，避免变成任意文件下载入口。
        if not path.is_relative_to(root) or path.suffix.lower() != ".md":
            return None
        if not path.is_file() or path.stat().st_size > MAX_MARKDOWN_BYTES:
            return None
        with path.open("rb") as stream:
            content = stream.read(MAX_MARKDOWN_BYTES + 1)
        if len(content) > MAX_MARKDOWN_BYTES:
            return None
        text = content.decode("utf-8")
        return text if text.strip() else None
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return None


def evidence_audit(summary: PaperSummary, source_text: str | None) -> list[dict]:
    """只验证引文是否出现在实际材料中，不验证模型结论、页码或因果关系。"""
    rows = []
    for index, evidence in enumerate(summary.evidence, 1):
        match = None
        if source_text and evidence.quote and evidence.paper_id == summary.paper_id:
            pattern = r"\s+".join(re.escape(word) for word in evidence.quote.split())
            match = re.search(pattern, source_text) if pattern else None
        row = {
            "序号": index,
            "章节": evidence.section or "未提供",
            "引文": evidence.quote or "",
            "核对": "原文匹配" if match else "未定位",
            "起始行": None,
            "结束行": None,
            "原文上下文": "",
        }
        if match:
            start = source_text.count("\n", 0, match.start()) + 1
            end = source_text.count("\n", 0, match.end()) + 1
            row.update(
                {
                    "起始行": start,
                    "结束行": end,
                    "原文上下文": source_text[max(0, match.start() - 120) : match.end() + 120],
                }
            )
        rows.append(row)
    return rows


def summary_to_markdown(
    summary: PaperSummary,
    paper: Paper | None = None,
    *,
    source_text: str | None = None,
) -> str:
    """将已生成的结构化总结转为 Markdown，保留材料来源与证据。"""
    title = paper.title if paper else str(summary.paper_id)
    source = "基于全文" if summary.generated_from_fulltext else "仅基于摘要降级生成"
    lines = [f"# {title} — 单篇论文总结", "", f"> 材料来源：{source}", ""]
    if paper:
        if paper.authors:
            lines.append(f"- 作者：{'、'.join(author.name for author in paper.authors)}")
        if paper.year:
            lines.append(f"- 年份：{paper.year}")
        if paper.doi:
            lines.append(f"- DOI：{paper.doi}")
        if paper.arxiv_id:
            lines.append(f"- arXiv：{paper.arxiv_id}")
    lines.extend([f"- 论文 ID：{summary.paper_id}", ""])
    sections = [
        ("核心问题", summary.core_problem),
        ("方法", summary.method),
        ("实验", summary.experiments),
        ("贡献", "\n".join(f"- {item}" for item in summary.contributions)),
        ("局限", "\n".join(f"- {item}" for item in summary.limitations)),
        ("与研究想法的相关性", summary.relevance_to_idea),
    ]
    for heading, text in sections:
        lines.extend([f"## {heading}", "", text.strip() or "材料中未明确说明。", ""])
    lines.extend(["## 证据引用", ""])
    for index, evidence in enumerate(summary.evidence, 1):
        location = [f"论文 {evidence.paper_id}"]
        if evidence.section:
            location.append(f"章节：{evidence.section}")
        if evidence.page:
            location.append(f"第 {evidence.page} 页")
        if evidence.markdown_anchor:
            location.append(f"锚点：{evidence.markdown_anchor}")
        lines.extend([f"{index}. {' · '.join(location)}", ""])
        if evidence.quote:
            lines.extend(f"> {line}" for line in evidence.quote.splitlines())
            lines.append("")
    if not summary.evidence:
        lines.extend(["未提供可定位的证据引用。", ""])
    if source_text is not None:
        lines.extend(
            ["## 原文引文核对", "", "> 仅核对文字是否出现在材料中，不代表结论或页码已被验证。", ""]
        )
        for row in evidence_audit(summary, source_text):
            location = f"，材料第 {row['起始行']}–{row['结束行']} 行" if row["起始行"] else ""
            lines.append(f"- 引文 {row['序号']}：{row['核对']}{location}")
        lines.append("")
    return "\n".join(lines)


def zip_markdown_files(files: Mapping[str, str]) -> bytes:
    """直接在内存中打包已准备好的 Markdown，不创建临时文件。"""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, markdown in files.items():
            archive.writestr(filename, markdown.encode("utf-8"))
    return buffer.getvalue()
