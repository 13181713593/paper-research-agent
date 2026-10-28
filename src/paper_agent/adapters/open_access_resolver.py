"""Group 3: locate a lawful open-access copy of a paper.

按成本优先逐渠道定位：已有候选/arXiv → Unpaywall → OpenAlex → 多个落地页。
``iter_candidate_batches`` 在当前候选下载失败后才查询下一渠道，成功即可停止。
Unpaywall 缺少 DOI/邮箱时跳过；OpenAlex 支持可选 OPENALEX_API_KEY。
``resolve`` 保留首批候选接口，供只需要 URL 的旧调用方使用。

The resolver never bypasses paywalls. ``is_open_access`` stays provider
metadata; ``fulltext.access_status`` records what this resolver actually
observed. A typed ``NoOpenAccessCopyError`` is raised when no lawful copy is
available, so the pipeline can fall back to the abstract or exclude the paper.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from contextlib import aclosing
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit

import httpx

from paper_agent.domain.errors import NoOpenAccessCopyError, ProviderUnavailableError
from paper_agent.domain.models import AccessStatus, Paper

UNPAYWALL_API = "https://api.unpaywall.org/v2"


def _http_url(value: str | None) -> str | None:
    """Return the value if it is a usable http(s) URL, otherwise ``None``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
    ):
        return None
    return value


def _looks_like_pdf(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


class UnpaywallResolver:
    """``OpenAccessResolver`` implementation built on direct links + Unpaywall."""

    def __init__(
        self,
        *,
        email: str | None = None,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        discover_locations: bool = False,
        openalex_api_key: str | None = None,
    ) -> None:
        # ``None`` means "read the environment"; "" explicitly disables the
        # Unpaywall lookup (deterministic for tests and offline runs).
        if email is None:
            email = os.getenv("UNPAYWALL_EMAIL") or ""
        self._email = email.strip()
        self._timeout = timeout_seconds
        self._transport = transport
        self._discover_locations = discover_locations
        self._openalex_api_key = (
            os.getenv("OPENALEX_API_KEY", "") if openalex_api_key is None else openalex_api_key
        ).strip()
        # 复用连接池：一次运行会对同一批站点发出大量短请求。
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def aclose(self) -> None:
        """Close the pooled client; safe to call when it was never opened."""
        client, self._client = self._client, None
        self._client_loop = None
        if client is not None:
            await client.aclose()

    def _client_for_loop(self) -> httpx.AsyncClient:
        """Return a pooled client bound to the current event loop."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=min(self._timeout, 15),
                max_redirects=5,
                transport=self._transport,
            )
            self._client_loop = loop
        return self._client

    async def resolve(self, paper: Paper) -> Paper:
        """兼容原接口：返回首批候选；下载编排器使用惰性分渠道接口。"""
        async with aclosing(self.iter_candidate_batches(paper)) as batches:
            async for urls in batches:
                if urls:
                    paper.fulltext.pdf_url = urls[0]
                    paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
                    return paper
        raise NoOpenAccessCopyError(f"no lawful open-access copy found for '{paper.title[:80]}'")

    async def iter_candidate_batches(self, paper: Paper) -> AsyncIterator[list[str]]:
        """兼容接口：只产出候选 URL，不标注取得渠道。"""
        async with aclosing(self.iter_candidate_batches_labelled(paper)) as batches:
            async for _channel, urls in batches:
                yield urls

    async def iter_candidate_batches_labelled(
        self, paper: Paper
    ) -> AsyncIterator[tuple[str, list[str]]]:
        """当前批次下载失败后才查询下一渠道；每次处理的状态均保存在本次迭代中。

        产出 ``(渠道, 候选 URL)``：渠道用于在界面上说明该篇文献的全文来自
        直链（arXiv/出版商）、Unpaywall、OpenAlex 还是落地页扫描。
        """
        attempt_start = len(paper.fulltext.attempts)
        found = False
        last_error = None
        self._direct_url(paper)
        urls = self._candidate_urls(paper)
        if urls:
            found = True
            yield "direct", urls
        landing_urls = [
            str(url)
            for url in (
                paper.fulltext.landing_page_url,
                paper.open_access_url,
                paper.landing_page_url,
                *(url for url in paper.fulltext.candidate_urls if not _looks_like_pdf(str(url))),
            )
            if url
        ]
        if self._email and paper.doi:
            try:
                before = set(self._candidate_urls(paper))
                await self._resolve_with_unpaywall(paper, landing_urls)
                self._record(
                    paper,
                    "unpaywall",
                    "succeeded",
                    http_status=200,
                    candidate_count=len(set(self._candidate_urls(paper)) - before),
                    reason="返回候选 URL，尚未验证 PDF 下载",
                )
                urls = self._candidate_urls(paper)
                if urls:
                    found = True
                    yield "unpaywall", urls
            except (NoOpenAccessCopyError, ProviderUnavailableError) as exc:
                last_error = exc
                self._record(paper, "unpaywall", "failed", error=exc)
        else:
            self._record(
                paper,
                "unpaywall",
                "skipped",
                reason="缺少 DOI" if not paper.doi else "未配置 UNPAYWALL_EMAIL",
            )
        if self._discover_locations:
            # 已失败副本的许可/版本不能沿用到来自其他仓储的新副本。
            paper.fulltext.license = None
            paper.fulltext.version = None
            await self._discover_openalex(paper, landing_urls)
            urls = self._candidate_urls(paper)
            if urls:
                found = True
                yield "openalex", urls
            # 聚合页不能代替已知 DOI 对应的出版社页面；已知 OA 落地页仍优先。
            if paper.doi:
                landing_urls.append(f"https://doi.org/{quote(paper.doi, safe='/')}")
            # 允许再扫描一层明确标为 PDF 的查看器页面，不跟随普通导航/参考文献。
            queue = [(url, 0) for url in dict.fromkeys(landing_urls)]
            visited: set[str] = set()
            while queue and len(visited) < 10:
                url, depth = queue.pop(0)
                if not _http_url(url) or url in visited:
                    continue
                visited.add(url)
                viewer_urls = await self._discover_landing_page(paper, url)
                urls = self._candidate_urls(paper)
                if urls:
                    found = True
                    yield "landing_page", urls
                if depth < 1:
                    queue.extend((viewer, depth + 1) for viewer in viewer_urls)
            if queue:
                self._record(
                    paper, "landing_page", "skipped", reason="达到单篇最多 10 个页面的定位预算"
                )
        if not found:
            if last_error is not None and not self._discover_locations:
                raise last_error
            unavailable = list(
                dict.fromkeys(
                    attempt["stage"]
                    for attempt in paper.fulltext.attempts[attempt_start:]
                    if attempt.get("error_kind") == "provider_unavailable"
                )
            )
            if unavailable:
                paper.fulltext.access_status = AccessStatus.UNAVAILABLE
                raise ProviderUnavailableError(
                    f"公开全文定位未完成：{', '.join(unavailable)} 请求失败或拒绝访问，详见获取尝试记录"
                )
            paper.fulltext.access_status = (
                AccessStatus.CLOSED if paper.is_open_access is False else AccessStatus.UNAVAILABLE
            )
            raise NoOpenAccessCopyError(
                f"no lawful open-access copy found for '{paper.title[:80]}'"
            )

    @staticmethod
    def _candidate_urls(paper: Paper) -> list[str]:
        urls = [paper.fulltext.pdf_url, *paper.fulltext.candidate_urls]
        cleaned = list(
            dict.fromkeys(value for url in urls if url and (value := _http_url(str(url))))
        )
        paper.fulltext.candidate_urls = cleaned
        return cleaned

    def _record(
        self,
        paper,
        stage,
        status,
        *,
        url=None,
        error=None,
        reason=None,
        http_status=None,
        candidate_count=None,
    ):
        attempt = {"stage": stage, "status": status}
        if url:
            attempt["url"] = url
        if error is not None:
            message = f"{type(error).__name__}: {error}"
            for secret in (self._openalex_api_key, self._email):
                if secret:
                    message = message.replace(secret, "[已隐藏]")
            attempt["error"] = message[:400]
            attempt["error_type"] = type(error).__name__
            if isinstance(error, httpx.HTTPStatusError):
                http_status = error.response.status_code
            if (
                isinstance(error, (httpx.TransportError, ProviderUnavailableError))
                or http_status in {401, 403, 408, 429}
                or (http_status and http_status >= 500)
            ):
                attempt["error_kind"] = "provider_unavailable"
            else:
                attempt["error_kind"] = "no_open_copy"
        if http_status is not None:
            attempt["http_status"] = str(http_status)
        if candidate_count is not None:
            attempt["candidate_count"] = str(candidate_count)
        if reason:
            attempt["reason"] = reason
        paper.fulltext.attempts.append(attempt)

    def _direct_url(self, paper: Paper) -> str | None:
        """Pick the first usable direct PDF URL from metadata, or ``None``."""
        candidates: list[str] = []
        pdf_url = _http_url(str(paper.fulltext.pdf_url)) if paper.fulltext.pdf_url else None
        if pdf_url:
            candidates.append(pdf_url)
        candidates.extend(str(url) for url in paper.fulltext.candidate_urls)
        oa_url = _http_url(str(paper.open_access_url)) if paper.open_access_url else None
        if oa_url:
            if _looks_like_pdf(oa_url):
                candidates.append(oa_url)
            elif not paper.fulltext.landing_page_url:
                # Provider OA link is a landing page; keep it for auditability.
                paper.fulltext.landing_page_url = oa_url
        if paper.arxiv_id:
            base_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
            if base_id:
                candidates.append(f"https://arxiv.org/pdf/{quote(base_id)}")
        for landing in [
            paper.landing_page_url,
            paper.open_access_url,
            *(r.source_url for r in paper.source_records),
        ]:
            if not landing:
                continue
            value = str(landing)
            if "arxiv.org/" in value:
                match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", value)
                if match:
                    candidates.append("https://arxiv.org/pdf/" + match[1].removesuffix(".pdf"))
            if "openaccess.thecvf.com/" in value and "/html/" in value:
                candidates.append(
                    value.replace("/html/", "/papers/").removesuffix(".html") + ".pdf"
                )
        paper.fulltext.candidate_urls = list(dict.fromkeys(candidates))
        return candidates[0] if candidates else None

    async def discover(self, paper: Paper) -> None:
        """兼容旧调用方，收集全部渠道；主流程按批次惰性查询，不会重复调用此方法。"""
        if not self._discover_locations:
            return
        try:
            async with aclosing(self.iter_candidate_batches(paper)) as batches:
                async for _ in batches:
                    pass
        except (NoOpenAccessCopyError, ProviderUnavailableError):
            # 旧 discover 契约只补充候选，未找到副本时由下载编排器决定降级。
            return

    async def _discover_openalex(self, paper: Paper, landing_urls: list[str]) -> None:
        identifier = (
            "https://doi.org/" + quote(paper.doi, safe="/")
            if paper.doi
            else quote(paper.openalex_id.rsplit("/", 1)[-1])
            if paper.openalex_id
            else None
        )
        if not identifier:
            self._record(paper, "openalex", "skipped", reason="缺少 DOI 或 OpenAlex ID")
            return
        url = "https://api.openalex.org/works/" + identifier
        # Key 只放在此 API 请求的 Header，不能传给后续出版社/仓储落地页。
        headers = (
            {"Authorization": f"Bearer {self._openalex_api_key}"} if self._openalex_api_key else {}
        )
        try:
            client = self._client_for_loop()
            response = await client.get(url, headers=headers, timeout=min(self._timeout, 15))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("OpenAlex 返回的正文不是对象")
            locations = [data.get("best_oa_location") or {}, *(data.get("locations") or [])]
            pdf_count = 0
            for location in locations:
                if not isinstance(location, dict) or location.get("is_oa") is False:
                    continue
                pdf_url = _http_url(location.get("pdf_url"))
                if pdf_url:
                    paper.fulltext.candidate_urls.append(pdf_url)
                    pdf_count += 1
                landing = _http_url(location.get("landing_page_url"))
                if landing and location.get("is_oa"):
                    landing_urls.append(landing)
                    if not paper.open_access_url:
                        paper.open_access_url = landing
            self._record(
                paper,
                "openalex",
                "succeeded",
                url=url,
                http_status=response.status_code,
                candidate_count=pdf_count,
                reason="元数据返回成功，但没有公开 PDF 直链"
                if not pdf_count
                else "发现公开 PDF 候选",
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self._record(paper, "openalex", "failed", url=url, error=exc)

    async def _discover_landing_page(self, paper: Paper, url: str) -> list[str]:
        viewer_urls = []
        candidate_count = 0
        try:
            client = self._client_for_loop()
            async with client.stream(
                "GET", url, timeout=min(self._timeout, 15), follow_redirects=True
            ) as response:
                response.raise_for_status()
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > 2_000_000:
                        raise ValueError("landing page exceeds 2 MB")
                if b"%PDF-" in content[:1024]:
                    paper.fulltext.candidate_urls.append(url)
                    candidate_count = 1
                else:
                    parser = _PdfLinks(str(response.url))
                    parser.feed(content.decode("utf-8", errors="replace"))
                    paper.fulltext.candidate_urls.extend(parser.urls[:5])
                    candidate_count = len(parser.urls[:5])
                    viewer_urls = parser.viewer_urls[:5]
            self._record(
                paper,
                "landing_page",
                "succeeded",
                url=url,
                http_status=response.status_code,
                candidate_count=candidate_count,
                reason="页面返回成功，但未发现 PDF 下载入口"
                if not candidate_count
                else "发现 PDF 下载入口候选",
            )
        except (httpx.HTTPError, ValueError) as exc:
            self._record(paper, "landing_page", "failed", url=url, error=exc)
        return viewer_urls

    async def _resolve_with_unpaywall(
        self,
        paper: Paper,
        landing_urls: list[str] | None = None,
    ) -> Paper:
        doi = (paper.doi or "").strip()
        url = f"{UNPAYWALL_API}/{quote(doi, safe='')}"
        try:
            client = self._client_for_loop()
            response = await client.get(url, params={"email": self._email}, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Unpaywall lookup failed for DOI {doi}: {exc}") from exc
        if response.status_code == 404:
            paper.fulltext.access_status = AccessStatus.UNAVAILABLE
            raise NoOpenAccessCopyError(f"Unpaywall has no record for DOI {doi}")
        if response.status_code != 200:
            raise ProviderUnavailableError(
                f"Unpaywall returned HTTP {response.status_code} for DOI {doi}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(f"Unpaywall returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderUnavailableError("Unpaywall 返回的正文不是对象")
        if not data.get("is_oa"):
            paper.fulltext.access_status = AccessStatus.CLOSED
            raise NoOpenAccessCopyError(f"Unpaywall reports no OA copy for DOI {doi}")
        candidate_locations = []
        if isinstance(data.get("best_oa_location"), dict):
            candidate_locations.append(data["best_oa_location"])
        other_locations = data.get("oa_locations") or []
        if not isinstance(other_locations, list):
            raise ProviderUnavailableError("Unpaywall oa_locations 格式错误")
        candidate_locations.extend(loc for loc in other_locations if isinstance(loc, dict))
        candidate_locations = [loc for loc in candidate_locations if loc.get("is_oa") is not False]
        if landing_urls is not None:
            for location in candidate_locations:
                landing = _http_url(location.get("url_for_landing_page")) or _http_url(
                    location.get("url")
                )
                if landing:
                    landing_urls.append(landing)
        paper.fulltext.candidate_urls.extend(
            loc["url_for_pdf"] for loc in candidate_locations if _http_url(loc.get("url_for_pdf"))
        )
        pdf_url = None
        chosen = {}
        for field in ("url_for_pdf", "url", "url_for_landing_page"):
            for location in candidate_locations:
                pdf_url = _http_url(location.get(field))
                if pdf_url:
                    chosen = location
                    break
            if pdf_url:
                break
        if pdf_url is None:
            paper.fulltext.access_status = AccessStatus.UNAVAILABLE
            raise NoOpenAccessCopyError(f"Unpaywall reports no usable OA URL for DOI {doi}")
        paper.fulltext.pdf_url = pdf_url
        paper.fulltext.access_status = AccessStatus.OPEN_ACCESS
        license_name = chosen.get("license")
        if isinstance(license_name, str) and license_name:
            paper.fulltext.license = license_name
        version = chosen.get("version")
        if isinstance(version, str) and version:
            paper.fulltext.version = version
        if not paper.fulltext.landing_page_url:
            landing = _http_url(chosen.get("url_for_landing_page"))
            if landing:
                paper.fulltext.landing_page_url = landing
        return paper


class _PdfLinks(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base = base_url
        self._priorities = {}
        self.viewer_urls = []
        self._anchor = None
        self._anchor_text = []
        self._anchor_pdf = False

    @property
    def urls(self):
        # 论文专用元标签/嵌入阅读器优先于站点上碰巧以 .pdf 结尾的附件。
        return sorted(self._priorities, key=self._priorities.get)

    def _add(self, value, *, priority=3, viewer=False):
        url = _http_url(urljoin(self.base, value)) if value else None
        if url:
            self._priorities[url] = min(priority, self._priorities.get(url, priority))
        if url and viewer and url not in self.viewer_urls:
            self.viewer_urls.append(url)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        mime = (attrs.get("type") or "").lower().split(";", 1)[0].strip()
        if tag == "meta" and (attrs.get("name") or "").lower() in {
            "citation_pdf_url",
            "wkhealth_pdf_url",
        }:
            self._add(attrs.get("content"), priority=0)
        elif tag in {"a", "link"}:
            href = attrs.get("href") or ""
            is_pdf = _looks_like_pdf(href) or mime == "application/pdf"
            if tag == "a":
                # 等到文字收齐再判断，避免先把投稿模板加入下载候选。
                self._anchor = href
                self._anchor_pdf = is_pdf
                self._anchor_text = [attrs.get("title") or "", attrs.get("aria-label") or ""]
            elif is_pdf:
                self._add(href)
        elif tag in {"iframe", "embed", "object"}:
            value = attrs.get("data") if tag == "object" else attrs.get("src")
            if value and (_looks_like_pdf(value) or mime == "application/pdf"):
                self._add(value, priority=1)

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor is not None:
            label = " ".join(" ".join(self._anchor_text).split()).lower()
            non_paper = re.search(
                r"\b(?:templates?|author guidelines?|submission guidelines?|instructions for authors)\b"
                r"|قالب|投稿模板|作者指南",
                label,
            )
            explicit_pdf = re.search(r"\bpdf\b", label)
            if not non_paper and (explicit_pdf or self._anchor_pdf):
                self._add(
                    self._anchor,
                    priority=2 if explicit_pdf else 3,
                    viewer=bool(explicit_pdf and not _looks_like_pdf(self._anchor)),
                )
            self._anchor = None
            self._anchor_text = []
            self._anchor_pdf = False
