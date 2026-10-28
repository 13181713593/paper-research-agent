"""为界面提供 Markdown 文件内容与内存打包，不重新调用生成接口。"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

from paper_agent.adapters.markdown_parser import IMAGE_LINK, MAX_IMAGE_BYTES, safe_asset_name
from paper_agent.domain.models import Paper, PaperSummary

MAX_MARKDOWN_BYTES = 20_000_000
MAX_PDF_BYTES = 30_000_000
MAX_BUNDLE_BYTES = 200_000_000


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


def load_paper_pdf(paper: Paper, data_dir: str | Path) -> bytes | None:
    """解析失败仍可下载原始 PDF，只读取数据目录内的有限大小文件。"""
    if not paper.fulltext.local_pdf_path:
        return None
    try:
        root = Path(data_dir).resolve()
        path = Path(paper.fulltext.local_pdf_path).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix.lower() != ".pdf" or not path.is_file():
            return None
        if path.stat().st_size > MAX_PDF_BYTES:
            return None
        with path.open("rb") as stream:
            content = stream.read(MAX_PDF_BYTES + 1)
        if len(content) > MAX_PDF_BYTES or b"%PDF-" not in content[:1024]:
            return None
        return content
    except (OSError, ValueError, RuntimeError):
        return None


def _load_image(paper: Paper, data_dir: str | Path, url: str) -> tuple[str, bytes] | None:
    name = safe_asset_name(url)
    if not name or not paper.fulltext.markdown_path:
        return None
    try:
        root = Path(data_dir).resolve()
        markdown = Path(paper.fulltext.markdown_path).resolve(strict=True)
        asset_dir = markdown.parent / f"{markdown.stem}_assets"
        path = (markdown.parent / name).resolve(strict=True)
        if not markdown.is_relative_to(root) or not path.is_relative_to(root):
            return None
        # 不能利用图片引用下载同目录的任意文件或其他论文的资源。
        if not path.is_relative_to(asset_dir) or not path.is_file():
            return None
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return None
        with path.open("rb") as stream:
            content = stream.read(MAX_IMAGE_BYTES + 1)
        if not content or len(content) > MAX_IMAGE_BYTES:
            return None
        return name, content
    except (OSError, ValueError, RuntimeError):
        return None


def embed_paper_images(paper: Paper, data_dir: str | Path, markdown: str) -> str:
    """预览和单文件下载使用内嵌图片，ZIP 则保留相对路径及资源文件。"""
    total = 0

    def replace(match: re.Match) -> str:
        nonlocal total
        image = _load_image(paper, data_dir, match.group("url"))
        if image is None:
            return "[图片不可用，请参照原始 PDF]"
        name, content = image
        total += len(content)
        if total > MAX_BUNDLE_BYTES:
            return "[图片过多，请下载含图片资源的 ZIP]"
        suffix = Path(name).suffix.lower()
        mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix[1:]}"
        url = f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"
        start, end = match.span("url")
        return match.group(0)[: start - match.start()] + url + match.group(0)[end - match.start() :]

    return IMAGE_LINK.sub(replace, markdown)


def paper_markdown_files(paper: Paper, data_dir: str | Path) -> dict[str, str | bytes]:
    """仅打包正文实际引用的安全图片，保持解压后的相对链接有效。"""
    markdown = load_paper_markdown(paper, data_dir)
    if markdown is None:
        return {}
    files: dict[str, str | bytes] = {}

    def replace(match: re.Match) -> str:
        image = _load_image(paper, data_dir, match.group("url"))
        if image is None:
            return "[图片不可用，请参照原始 PDF]"
        name, content = image
        files[name] = content
        return match.group(0)

    files[markdown_filename(paper.title, paper.paper_id, "paper")] = IMAGE_LINK.sub(
        replace, markdown
    )
    return files


def zip_paper_pdfs(papers: Sequence[Paper], data_dir: str | Path) -> bytes:
    """按论文逐个读取和写入 ZIP，避免整批 PDF 同时驻留内存。"""
    buffer = BytesIO()
    total = 0
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for paper in papers:
            content = load_paper_pdf(paper, data_dir)
            if content is None:
                continue
            total += len(content)
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("PDF 总大小超过 200 MB，请减少本批数量或分别下载。")
            filename = markdown_filename(paper.title, paper.paper_id, "paper")
            archive.writestr(str(Path(filename).with_suffix(".pdf")), content)
    return buffer.getvalue() if total else b""


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


def zip_markdown_files(files: Mapping[str, str | bytes]) -> bytes:
    """在内存中打包 Markdown 和图片，限制总大小并拒绝越界条目名。"""
    buffer = BytesIO()
    total = 0
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts or "\\" in filename or ":" in filename:
                raise ValueError("压缩包文件名不安全。")
            data = content.encode("utf-8") if isinstance(content, str) else content
            total += len(data)
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("Markdown 和图片总大小超过 200 MB，请分篇下载。")
            archive.writestr(filename, data)
    return buffer.getvalue()
