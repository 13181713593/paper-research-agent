"""Browser search hits are discovery records, not verified paper abstracts."""

import base64
import re
from datetime import date
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from pydantic import Field, HttpUrl

from paper_agent.domain.models import Paper, PaperSource, SourceRecord, StrictModel


def clean_result_url(url):
    parsed = urlsplit(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        url = parse_qs(parsed.query).get("uddg", [url])[0]
        parsed = urlsplit(url)
    if parsed.hostname in {"bing.com", "www.bing.com"} and parsed.path.startswith("/ck/a"):
        value = parse_qs(parsed.query).get("u", [""])[0]
        if value.startswith("a1"):
            try:
                encoded = value[2:]
                url = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
                parsed = urlsplit(url)
            except (ValueError, UnicodeError):
                return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def is_paper_url(url):
    parsed = urlsplit(str(url))
    host = (parsed.hostname or "").lower().removeprefix("www.")
    paths = {
        "arxiv.org": ("/abs/", "/pdf/", "/html/"),
        "doi.org": ("/10.",),
        "aclanthology.org": ("/20", "/P", "/N", "/D", "/W"),
        "openreview.net": ("/forum", "/pdf"),
        "pubmed.ncbi.nlm.nih.gov": ("/",),
        "pmc.ncbi.nlm.nih.gov": ("/articles/",),
        "ncbi.nlm.nih.gov": ("/pmc/articles/",),
        "semanticscholar.org": ("/paper/",),
        "nature.com": ("/articles/",),
        "sciencedirect.com": ("/science/article/",),
        "link.springer.com": ("/article/", "/chapter/"),
        "ieeexplore.ieee.org": ("/document/",),
        "dl.acm.org": ("/doi/",),
        "biorxiv.org": ("/content/",),
        "medrxiv.org": ("/content/",),
        "frontiersin.org": ("/journals/", "/articles/"),
        "onlinelibrary.wiley.com": ("/doi/",),
        "academic.oup.com": ("/",),
    }
    return bool(host in paths and any(parsed.path.startswith(p) for p in paths[host]))


class BrowserSearchHit(StrictModel):
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    snippet: str = Field(default="", max_length=2000)
    engine: str
    query: str
    publication_year: int | None = Field(default=None, ge=1600, le=2200)
    metadata_verified: bool = False
    abstract: str | None = None
    publication_date: date | None = None
    authors: list[str] = Field(default_factory=list)
    pdf_url: HttpUrl | None = None
    metadata_source: str | None = None

    @property
    def discovery_identity(self):
        url = str(self.url)
        arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^?#]+)", url)
        if arxiv:
            identifier = re.sub(r"v\d+$", "", arxiv[1].removesuffix(".pdf"))
            return "arxiv:" + identifier.casefold()
        doi = re.search(r"10\.\d{4,9}/[^\s?#]+", unquote(url))
        return "doi:" + doi[0].casefold() if doi else url

    @property
    def paper_candidate(self):
        return is_paper_url(self.url) or (
            self.metadata_verified and (self.metadata_source or "").endswith(" API")
        )

    def search_context(self):
        if self.metadata_verified and self.abstract:
            return (
                f"题名：{self.title}\n来源：{self.url}\n"
                f"论文摘要（来自{self.metadata_source or '论文元数据'}，未核验全文）：{self.abstract[:1500]}"
            )
        return (
            f"题名：{self.title}\n来源：{self.url}\n检索片段（未经全文核验）：{self.snippet[:1000]}"
        )

    def to_paper(self):
        if not self.paper_candidate:
            raise ValueError("该网址未确认为论文详情页，可作为检索上下文使用")
        url = str(self.url)
        arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^?#]+)", url)
        doi = re.search(r"10\.\d{4,9}/[^\s?#]+", unquote(url))
        return Paper(
            title=self.title,
            abstract=self.abstract if self.metadata_verified else None,
            year=self.publication_year,
            publication_date=self.publication_date,
            authors=[{"name": name} for name in self.authors],
            landing_page_url=url,
            open_access_url=self.pdf_url,
            arxiv_id=arxiv[1].removesuffix(".pdf") if arxiv else None,
            doi=doi[0] if doi else None,
            source_records=[
                SourceRecord(
                    source=PaperSource.BROWSER,
                    source_id=url,
                    source_url=url,
                    raw_metadata={
                        "engine": self.engine,
                        "query": self.query,
                        "search_snippet": self.snippet,
                        "metadata_verified": self.metadata_verified,
                        "metadata_source": self.metadata_source,
                    },
                )
            ],
        )


class BrowserSearchResults(StrictModel):
    hits: list[BrowserSearchHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    duration_ms: int = 0
