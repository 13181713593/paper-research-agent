"""Real browser discovery with per-run contexts and a Windows subprocess-safe loop."""

import asyncio
import math
import os
import re
import sys
from datetime import UTC, date, datetime
from time import perf_counter
from urllib.parse import unquote, urlencode, urljoin, urlsplit

from defusedxml import ElementTree

from paper_agent.adapters._concept_extraction import (
    _SYNONYM_GROUPS,
    _ZH_EN_TERMS,
    HeuristicConceptExtractor,
)
from paper_agent.adapters._retriever_utils import http_get_json
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.metadata_relevance import MetadataRelevance
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.domain.browser_search import (
    BrowserSearchHit,
    BrowserSearchResults,
    clean_result_url,
)
from paper_agent.domain.models import ResearchIdea, SearchPlan, SearchQuery
from paper_agent.domain.publication_policy import years_before

ENGINES = {
    "Bing": ("https://www.bing.com/search", "li.b_algo", "h2 a", ".b_caption p"),
    "arXiv网页": ("https://arxiv.org/search/", "li.arxiv-result", "p.list-title a", "p.abstract"),
    "PubMed网页": (
        "https://pubmed.ncbi.nlm.nih.gov/",
        ".docsum-content",
        "a.docsum-title",
        ".docsum-snippet",
    ),
    "DuckDuckGo": (
        "https://html.duckduckgo.com/html/",
        ".result",
        ".result__a",
        ".result__snippet",
    ),
}


def academic_query(query):
    """Keep recognized academic phrases intact and restrict discovery to scholarly sites."""
    phrases = {*_ZH_EN_TERMS.values(), *(term for group in _SYNONYM_GROUPS for term in group)}
    if '"' not in query:
        matches = [
            p
            for p in phrases
            if len(p.split()) >= 2 and re.search(r"\b" + re.escape(p) + r"\b", query, re.IGNORECASE)
        ]
        if matches:
            phrase = max(matches, key=len)
            query = re.sub(
                r"\b" + re.escape(phrase) + r"\b",
                lambda m: '"' + m[0] + '"',
                query,
                count=1,
                flags=re.IGNORECASE,
            )
    query = re.sub(r"\s+(?:papers?|research)\s*$", "", query, flags=re.IGNORECASE)
    query = " ".join(query.split())
    if "site:" not in query.casefold():
        query += " site:semanticscholar.org"
    return query


def relevant_discovery_hits(hits, query):
    words = set(re.findall(r"[a-z]{3,}", query.casefold())) - {
        "paper",
        "papers",
        "research",
        "study",
        "with",
        "and",
        "for",
        "the",
        "site",
        "org",
        "com",
    }
    if len(words) < 2:
        return hits
    return [
        hit
        for hit in hits
        if len(words & set(re.findall(r"[a-z]{3,}", (hit.title + " " + hit.snippet).casefold())))
        >= 2
    ]


def browser_search_url(engine, query):
    if engine in {"arXiv网页", "PubMed网页"}:
        cleaned = re.sub(r"\s+(?:papers?|research)\s*$", "", query, flags=re.IGNORECASE)
        params = (
            {
                "query": cleaned.strip(),
                "searchtype": "all",
                "abstracts": "show",
                "order": "-announced_date_first",
                "size": "25",
            }
            if engine == "arXiv网页"
            else {"term": cleaned.strip(), "sort": "date"}
        )
        return ENGINES[engine][0] + "?" + urlencode(params)
    return ENGINES[engine][0] + "?" + urlencode({"q": academic_query(query)})


def parse_bing_feed(xml, query, limit):
    """Bing's public search feed avoids depending on JavaScript-rendered cards."""
    root = ElementTree.fromstring(xml)
    hits = []
    for item in root.findall("./channel/item"):
        url = clean_result_url(item.findtext("link", ""))
        title = " ".join(item.findtext("title", "").split())[:500]
        if title and url:
            hits.append(
                BrowserSearchHit(
                    title=title,
                    url=url,
                    engine="Bing",
                    query=query,
                    snippet=item.findtext("description", "")[:2000],
                )
            )
        if len(hits) >= limit:
            break
    return hits


class BrowserPaperSearch:
    def __init__(self, *, channel=None, timeout=10, concurrency=3, database_fallback=True):
        self.channel = (
            channel
            or os.getenv("PAPER_AGENT_BROWSER_CHANNEL")
            or ("msedge" if sys.platform == "win32" else "chromium")
        )
        self.timeout = timeout
        self.concurrency = concurrency
        self.database_fallback = database_fallback

    async def search(self, queries, *, engines=("Bing",), limit=10):
        queries = list(dict.fromkeys(q.strip()[:500] for q in queries if q.strip()))[:4]
        if not queries:
            raise ValueError("请输入浏览器检索词")
        if not engines or any(engine not in ENGINES for engine in engines):
            raise ValueError("请选择支持的网页论文检索引擎")
        # Never share Playwright objects between Streamlit/asyncio threads.
        return await asyncio.to_thread(self._run, queries, tuple(engines), max(1, min(limit, 30)))

    def _run(self, queries, engines, limit):
        # Streamlit may use SelectorEventLoop on Windows, which cannot create
        # Playwright subprocesses. Keep a Proactor loop local to this worker.
        factory = asyncio.ProactorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(self._search(queries, engines, limit))

    async def _search(self, queries, engines, limit):
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                '请安装项目依赖：python -m pip install -e ".[dev,processing]"'
            ) from exc
        started = perf_counter()
        result = BrowserSearchResults(
            queries=[browser_search_url(e, q) for e in engines for q in queries]
        )
        async with async_playwright() as playwright:
            try:
                launch_options = {}
                proxy_url = (
                    os.getenv("PAPER_AGENT_BROWSER_PROXY")
                    or os.getenv("HTTPS_PROXY")
                    or os.getenv("HTTP_PROXY")
                )
                if proxy_url:
                    proxy = urlsplit(proxy_url)
                    launch_options["proxy"] = {
                        "server": f"{proxy.scheme}://{proxy.hostname}:{proxy.port or 80}"
                    }
                    if proxy.username:
                        launch_options["proxy"]["username"] = unquote(proxy.username)
                    if proxy.password:
                        launch_options["proxy"]["password"] = unquote(proxy.password)
                browser = await playwright.chromium.launch(
                    headless=True,
                    **launch_options,
                    **({"channel": self.channel} if self.channel != "chromium" else {}),
                )
            except Exception as exc:
                raise RuntimeError(
                    "浏览器未能启动。Windows 请安装 Edge；其他系统运行 python -m playwright install chromium。"
                ) from exc
            try:
                context = await browser.new_context(locale="zh-CN")
                await context.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in {"image", "media", "font"}
                        else route.continue_()
                    ),
                )
                gate = asyncio.Semaphore(self.concurrency)
                fallback_task = (
                    asyncio.create_task(self._database_results(queries, limit))
                    if self.database_fallback
                    else None
                )

                async def one(engine, query):
                    async with gate:
                        page = await context.new_page()
                        try:
                            return await self._search_page(page, engine, query, limit), None
                        except Exception as exc:  # noqa: BLE001 - per-engine isolation
                            return (
                                [],
                                f"{engine}：{type(exc).__name__}，{str(exc).splitlines()[0][:180]}",
                            )
                        finally:
                            await page.close()

                batches = await asyncio.gather(
                    *(one(engine, query) for engine in engines for query in queries)
                )
                seen = set()
                for hits, warning in batches:
                    if warning:
                        result.warnings.append(warning)
                    for hit in hits:
                        if hit.discovery_identity not in seen:
                            seen.add(hit.discovery_identity)
                            result.hits.append(hit)
                result.hits = result.hits[:limit]

                async def enrich(hit):
                    if not hit.paper_candidate:
                        return
                    async with gate:
                        page = await context.new_page()
                        try:
                            if not await self._enrich_doi(hit):
                                await self._enrich_page(page, hit)
                        except Exception as exc:  # noqa: BLE001 - preserve the hit on metadata failure
                            result.warnings.append(
                                f"论文详情元数据未补全：{hit.title[:60]}（{type(exc).__name__}）"
                            )
                        finally:
                            await page.close()

                await asyncio.gather(*(enrich(hit) for hit in result.hits[:5]))
                fallback, fallback_warnings = await fallback_task if fallback_task else ([], [])
                if not any(hit.paper_candidate for hit in result.hits) and fallback:
                    result.hits = [*fallback, *result.hits][:limit]
                    result.warnings.append(
                        "网页引擎未返回可确认的主题相关论文；已使用并发查询的论文数据库补充，实际来源见结果表。"
                    )
                    result.warnings.extend(fallback_warnings)
                if not result.hits and not result.warnings:
                    result.warnings.append(
                        "搜索页面未返回可解析结果；可点击搜索引擎链接在浏览器中查看。"
                    )
            finally:
                await browser.close()
        result.duration_ms = int((perf_counter() - started) * 1000)
        return result

    async def _search_page(self, page, engine, query, limit):
        if engine == "Bing":
            response = await page.goto(
                browser_search_url(engine, query) + "&format=rss",
                wait_until="domcontentloaded",
                timeout=int(self.timeout * 1000),
            )
            if response and response.status < 400:
                try:
                    hits = relevant_discovery_hits(
                        parse_bing_feed(await response.text(), academic_query(query), limit), query
                    )
                    return hits
                except ElementTree.ParseError:
                    pass
        _, rows, heading, snippet = ENGINES[engine]
        response = await page.goto(
            browser_search_url(engine, query),
            wait_until="domcontentloaded",
            timeout=int(self.timeout * 1000),
        )
        if response and response.status >= 400:
            raise RuntimeError(f"搜索引擎返回 HTTP {response.status}")
        # Visible challenge pages are reported; never solve or evade a CAPTCHA.
        if await page.locator('input[name="captcha"], iframe[src*="captcha"], #b_captcha').count():
            raise RuntimeError("搜索引擎要求人工验证，请在浏览器中处理")
        # Bing can insert empty result containers before hydrating their titles.
        # Waiting for the container alone can incorrectly report zero results.
        try:
            await page.locator(f"{rows} {heading}").first.wait_for(
                state="visible", timeout=int(self.timeout * 1000)
            )
        except Exception as exc:
            text = " ".join((await page.locator("body").inner_text(timeout=2000)).split())
            raise RuntimeError(f"搜索页面未显示结果：{await page.title()}；{text[:220]}") from exc
        hits = []
        for row in await page.locator(rows).all():
            link = row.locator(heading).first
            if not await link.count():
                continue
            url = clean_result_url(urljoin(page.url, await link.get_attribute("href") or ""))
            title_locator = row.locator("p.title").first if engine == "arXiv网页" else link
            title = " ".join((await title_locator.inner_text()).split())[:500]
            if not url or not title:
                continue
            desc = row.locator(snippet).first
            summary = (
                " ".join((await desc.inner_text()).split())[:2000] if await desc.count() else ""
            )
            hits.append(
                BrowserSearchHit(title=title, url=url, snippet=summary, engine=engine, query=query)
            )
            if len(hits) >= limit:
                break
        return relevant_discovery_hits(hits, query)

    async def _enrich_page(self, page, hit):
        if str(hit.url).split("?", 1)[0].endswith(".pdf") or urlsplit(str(hit.url)).path.startswith(
            "/pdf/"
        ):
            return
        response = await page.goto(
            str(hit.url), wait_until="domcontentloaded", timeout=int(min(8, self.timeout) * 1000)
        )
        if response and response.status >= 400:
            return

        async def values(name):
            locator = page.locator(f'meta[name="{name}" i]')
            return [
                value
                for value in [await item.get_attribute("content") for item in await locator.all()]
                if value
            ]

        titles = await values("citation_title") or await values("DC.title")
        if not titles:
            return
        hit.title = titles[0][:500]
        hit.metadata_verified = True
        hit.metadata_source = "论文页面 citation 元数据"
        hit.authors = (await values("citation_author"))[:30]
        dates = (
            await values("citation_publication_date")
            or await values("citation_date")
            or await values("DC.date")
        )
        if dates:
            match = re.match(r"^(\d{4})(?:[-/](\d{1,2})[-/](\d{1,2}))?", dates[0])
            if match:
                hit.publication_year = int(match[1])
                if match[2] and match[3]:
                    try:
                        hit.publication_date = date(int(match[1]), int(match[2]), int(match[3]))
                    except ValueError:
                        pass
        abstracts = await values("citation_abstract") or await values("DC.description")
        if abstracts:
            hit.abstract = abstracts[0][:12000]
        pdfs = await values("citation_pdf_url")
        if pdfs and clean_result_url(pdfs[0]):
            hit.pdf_url = pdfs[0]

    async def _enrich_doi(self, hit):
        match = re.search(r"10\.\d{4,9}/[^\s?#]+", unquote(str(hit.url)))
        if not match:
            return False
        try:
            payload = await asyncio.to_thread(
                http_get_json,
                "https://api.crossref.org/works/" + match[0],
                timeout=6,
                max_retries=0,
            )
        except Exception:  # noqa: BLE001 - optional publisher metadata supplement
            return False
        paper = CrossrefRetriever()._to_paper(payload.get("message", {}))
        if paper is None:
            return False
        hit.title = paper.title[:500]
        hit.publication_year = paper.year
        hit.publication_date = paper.publication_date
        hit.authors = [author.name for author in paper.authors]
        hit.abstract = paper.abstract
        hit.metadata_verified = True
        hit.metadata_source = "Crossref DOI 记录"
        return True

    async def _database_results(self, queries, limit):
        """Run free metadata providers in parallel; never label their output as a web hit."""
        today = datetime.now(UTC).date()
        gate = asyncio.Semaphore(4)
        adapters = [
            CrossrefRetriever(timeout=8),
            OpenAlexRetriever(timeout=8),
            SemanticScholarRetriever(timeout=8),
        ]

        async def one(adapter, raw):
            requested_years = [int(y) for y in re.findall(r"\b(20\d{2})\b", raw)]
            lower = date(min(requested_years), 1, 1) if requested_years else years_before(today, 5)
            upper = min(today, date(max(requested_years), 12, 31)) if requested_years else today
            cleaned = re.sub(r"\b20\d{2}\b", " ", raw)
            cleaned = re.sub(r"\s+(?:papers?|research)\s*$", "", cleaned, flags=re.IGNORECASE)
            local = HeuristicConceptExtractor().extract(raw)
            english = [c for c in local.concepts if not re.search(r"[\u4e00-\u9fff]", c)]
            if re.search(r"[\u4e00-\u9fff]", raw) and english:
                cleaned = " ".join(english)
            query = SearchQuery(
                source=adapter.source_name,
                query=" ".join(cleaned.split()),
                limit=limit,
                filters={
                    "date_from": lower.isoformat(),
                    "date_to": upper.isoformat(),
                },
            )
            try:
                async with gate:
                    papers = await adapter.search(SearchPlan(intent_summary=raw, queries=[query]))
                relevance = MetadataRelevance(ResearchIdea(text=cleaned), papers)
                start = lower
                papers = [
                    p
                    for p in papers
                    if relevance.score(p)[2]
                    and (not p.publication_date or start <= p.publication_date <= upper)
                    and (p.year is None or start.year <= p.year <= upper.year)
                ]
                return papers, adapter.source_name, query.query, None
            except Exception as exc:  # noqa: BLE001 - optional provider isolation
                return (
                    [],
                    adapter.source_name,
                    query.query,
                    f"数据库补充 {adapter.source_name}：{type(exc).__name__}",
                )

        batches = await asyncio.gather(
            *(one(adapter, query) for query in queries for adapter in adapters)
        )
        hits, warnings, seen = [], [], set()
        for papers, source, actual_query, warning in batches:
            if warning:
                warnings.append(warning)
            for paper in papers:
                url = paper.landing_page_url or (
                    f"https://doi.org/{paper.doi}" if paper.doi else None
                )
                if not url:
                    continue
                hit = BrowserSearchHit(
                    title=paper.title[:500],
                    url=url,
                    engine=f"{source} 数据库补充",
                    query=actual_query,
                    snippet=(paper.abstract or "")[:2000],
                    abstract=paper.abstract,
                    publication_year=paper.year,
                    publication_date=paper.publication_date,
                    authors=[a.name for a in paper.authors],
                    metadata_verified=True,
                    metadata_source=f"{source} API",
                    pdf_url=paper.open_access_url,
                )
                if hit.discovery_identity not in seen:
                    hits.append(hit)
                    seen.add(hit.discovery_identity)
        cutoff = years_before(today, 2)
        recent = [
            h
            for h in hits
            if (h.publication_date and h.publication_date >= cutoff)
            or (h.publication_year and h.publication_year > cutoff.year)
        ]
        selected = {h.discovery_identity for h in recent[: math.ceil(limit * 0.7)]}
        for hit in hits:
            if len(selected) >= limit:
                break
            selected.add(hit.discovery_identity)
        return [h for h in hits if h.discovery_identity in selected][:limit], warnings
