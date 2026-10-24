"""按指定编号补全 arXiv 元数据，不执行主题检索或相关性排序。"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlsplit

import httpx
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import Author, Paper, PaperSource, PublicationStatus, SourceRecord

ARXIV_ID = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v[1-9]\d*)?", re.IGNORECASE
)
_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_MAX_RESPONSE_BYTES = 2_000_000


def _text(element: ET.Element, path: str) -> str:
    return " ".join((element.findtext(path, default="", namespaces=_NS) or "").split())


def _base_id(identifier: str) -> str:
    return re.sub(r"v\d+$", "", identifier).lower()


def _entry_id(entry: ET.Element) -> str | None:
    url = urlsplit(_text(entry, "atom:id"))
    if url.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        return None
    if not url.path.startswith("/abs/"):
        return None
    identifier = url.path.removeprefix("/abs/")
    return identifier if ARXIV_ID.fullmatch(identifier) else None


class ArxivMetadataEnricher:
    """单次批量查询，按编号对齐响应；接口失败不妨碍后续全文处理。"""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._transport = transport

    async def enrich(self, papers: list[Paper]) -> list[str]:
        """更新工作台的论文副本并返回提示；不凭模型知识补写缺失字段。"""
        targets = [(index, p) for index, p in enumerate(papers) if p.arxiv_id]
        if not targets:
            return []
        identifiers = list(dict.fromkeys(p.arxiv_id for _, p in targets))
        if len(identifiers) > 20 or any(not ARXIV_ID.fullmatch(i) for i in identifiers):
            raise ValueError("元数据查询仅接受最多 20 个有效 arXiv 编号。")
        try:
            entries = await self._fetch(identifiers)
        except (
            httpx.HTTPError,
            ET.ParseError,
            DefusedXmlException,
            ProviderUnavailableError,
        ) as exc:
            return [
                (
                    f"arXiv 基本信息获取失败（{type(exc).__name__}）；继续下载与总结，"
                    "缺失字段不表示论文中没有这些信息。"
                )
            ]
        warnings = []
        for index, paper in targets:
            requested = paper.arxiv_id
            candidates = [
                (identifier, entry)
                for identifier, entry in entries
                if _base_id(identifier) == _base_id(requested)
                and (not re.search(r"v\d+$", requested) or identifier.lower() == requested.lower())
            ]
            if not candidates:
                warnings.append(f"arXiv:{requested} 未返回匹配版本的基本信息；继续处理全文。")
                continue
            # 无版本编号使用返回的最高版本；有版本编号必须精确匹配。
            identifier, entry = max(
                candidates,
                key=lambda item: (
                    int(re.search(r"v(\d+)$", item[0])[1]) if re.search(r"v(\d+)$", item[0]) else 0
                ),
            )
            try:
                papers[index] = self._merge(paper, identifier, entry)
            except ValueError:
                warnings.append(f"arXiv:{requested} 基本信息格式异常；保留原信息并继续处理。")
        return warnings

    async def _fetch(self, identifiers: list[str]) -> list[tuple[str, ET.Element]]:
        async with (
            httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
                headers={"User-Agent": "paper-research-agent/0.2 (metadata lookup)"},
            ) as client,
            client.stream(
                "GET",
                _API,
                params={"id_list": ",".join(identifiers), "max_results": len(identifiers)},
            ) as response,
        ):
            response.raise_for_status()
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > _MAX_RESPONSE_BYTES:
                    raise ProviderUnavailableError("arXiv 元数据响应过大")
        root = SafeET.fromstring(bytes(content), forbid_dtd=True)
        if root.tag != f"{{{_NS['atom']}}}feed":
            raise ProviderUnavailableError("arXiv 未返回 Atom 元数据")
        return [
            (identifier, entry)
            for entry in root.findall("atom:entry", _NS)
            if (identifier := _entry_id(entry)) is not None
        ]

    @staticmethod
    def _merge(paper: Paper, identifier: str, entry: ET.Element) -> Paper:
        title = _text(entry, "atom:title")
        if not title:
            raise ValueError("arXiv 元数据缺少题名")
        authors = [
            Author(
                name=name,
                affiliations=list(
                    dict.fromkeys(
                        " ".join((aff.text or "").split())
                        for aff in node.findall("arxiv:affiliation", _NS)
                        if (aff.text or "").strip()
                    )
                ),
            )
            for node in entry.findall("atom:author", _NS)
            if (name := _text(node, "atom:name"))
        ]
        abstract = _text(entry, "atom:summary")
        published = _text(entry, "atom:published")
        landing = f"https://arxiv.org/abs/{identifier}"
        pdf = f"https://arxiv.org/pdf/{identifier}"
        data = paper.model_dump()
        data.update(title=title, landing_page_url=landing, is_open_access=True, open_access_url=pdf)
        if authors:
            data["authors"] = authors
            data["institutions"] = list(
                dict.fromkeys(
                    [
                        *paper.institutions,
                        *(aff for author in authors for aff in author.affiliations),
                    ]
                )
            )
        if abstract:
            data["abstract"] = abstract
        if published:
            published_date = datetime.fromisoformat(published).date()
            data.update(publication_date=published_date, year=published_date.year)
        if paper.publication_status == PublicationStatus.UNKNOWN:
            data["publication_status"] = PublicationStatus.PREPRINT
        for field, path in (("doi", "arxiv:doi"), ("comments", "arxiv:comment")):
            if value := _text(entry, path):
                data[field] = value
        data["source_records"] = [
            r
            for r in paper.source_records
            if not (r.source == PaperSource.ARXIV and _base_id(r.source_id) == _base_id(identifier))
        ] + [
            SourceRecord(
                source=PaperSource.ARXIV,
                source_id=identifier,
                source_url=landing,
                raw_metadata={
                    "metadata_method": "arxiv_atom_api",
                    "title": title,
                    "authors": [author.model_dump() for author in authors],
                    "abstract": abstract or None,
                    "published": published or None,
                    "journal_ref": _text(entry, "arxiv:journal_ref") or None,
                },
            )
        ]
        data["fulltext"].update(
            pdf_url=pdf,
            landing_page_url=landing,
            version=(match[0] if (match := re.search(r"v\d+$", identifier)) else None),
        )
        return Paper.model_validate(data)
