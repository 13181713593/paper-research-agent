"""Offline reproduction of the four retrieval issues under review.

Each check is deterministic and does NOT hit the network by default; it stubs
the provider HTTP layer so the behaviour under review is isolated. Pass
``--live`` to additionally probe the real Semantic Scholar bulk endpoint and
the arXiv API (slow, rate limited, may 429).

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_retrieval_issues.py
    .venv\\Scripts\\python.exe scripts\\verify_retrieval_issues.py --live

Exit code 0 = none of the four issues reproduced; 1 = at least one reproduced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_agent.adapters import semantic_scholar_retriever as s2_module
from paper_agent.adapters.arxiv_retriever import ArxivRetriever
from paper_agent.adapters.openreview_retriever import OpenReviewRetriever
from paper_agent.adapters.paper_merger import PaperMerger
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.retrieval_common import normalize_title
from paper_agent.domain.models import (
    Author,
    Paper,
    PaperSource,
    ResearchIdea,
    SearchPlan,
    SearchQuery,
)

IDEA = "边缘计算环境下的任务卸载与资源调度"


def _title(text: str) -> None:
    print(f"\n== {text} ==")


def _verdict(hit: bool) -> None:
    print(f"   [结论] {'命中问题' if hit else '未命中（行为符合预期）'}")


# ---------------------------------------------------------------------------
# 1. Semantic Scholar: publication-date sort silently switches to /bulk, which
#    ignores `limit` and returns one fixed server batch (up to 1000 records).
# ---------------------------------------------------------------------------
def check_s2_bulk() -> bool:
    _title("问题1：Semantic Scholar 按时间排序时走 /bulk，limit 失效")

    captured: dict = {}

    def fake_http(url, *, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = dict(params or {})
        # The relevance endpoint truncates server-side to `limit`; the bug we
        # detect is the bulk endpoint, which returns a fixed batch and ignores
        # `limit`. Emulate a well-behaved server here.
        limit = int(captured["params"].get("limit", 100))
        abstract = "a" * 1200
        payload = {
            "data": [
                {"paperId": f"p{i}", "title": f"Paper {i}", "abstract": abstract}
                for i in range(limit)
            ],
            "total": limit,
        }
        captured["bytes"] = len(json.dumps(payload).encode())
        return payload

    original = s2_module.http_get_json
    s2_module.http_get_json = fake_http
    try:
        retriever = s2_module.SemanticScholarRetriever()
        papers = retriever._to_papers(
            retriever._fetch(
                "large language model agents",
                limit=7,
                filters={"date_from": "2024-10-27", "date_to": "2026-10-27"},
                sort_by="publication_date",
            )
        )
    finally:
        s2_module.http_get_json = original

    url_is_bulk = captured["url"].endswith("/bulk")
    has_limit = "limit" in captured["params"]
    print(f"   请求 URL          : {captured['url']}")
    print(f"   请求参数          : {sorted(captured['params'])}")
    print(f"   是否走 /bulk      : {url_is_bulk}")
    print(f"   limit 参数是否存在: {has_limit}")
    print(f"   返回条数          : {len(papers)} 条（参数 limit=7）")
    hit = url_is_bulk and not has_limit
    print("   [判定] 走 /bulk 且无 limit 参数 => 每条 query 在下载整批记录而非按需截断")
    _verdict(hit)
    return hit


# ---------------------------------------------------------------------------
# 2. arXiv: the planner's multi-phrase variants are joined with AND, which the
#    planner's own comment says collapses recall (3 phrases -> 0 hits).
# ---------------------------------------------------------------------------
def check_arxiv_and() -> bool:
    _title("问题2：arXiv 组合式 AND 坍缩 / 只取首短语丢失场景词")

    idea = ResearchIdea(text=IDEA)
    plan = asyncio.run(HeuristicQueryPlanner(expand_queries=True).plan(idea, 30))
    arxiv_queries = [q for q in plan.queries if q.source == PaperSource.ARXIV]
    retriever = ArxivRetriever()

    print(f"   planner 生成的概念 : {plan.concepts[:3]}")
    expressions = [retriever._build_query(q) for q in arxiv_queries]
    for expression in dict.fromkeys(expressions):
        print(f"   {expression}")

    # 1) 不能出现多短语 AND（召回坍缩）。
    hit_and = any(expression.count(" AND all:") >= 2 for expression in expressions)
    # 2) 核心概念不能因「只取首短语」被丢弃（如 edge computing 场景限定词）。
    built_text = " ".join(expressions).casefold()
    missing = [c for c in plan.concepts if c.strip() and c.casefold() not in built_text]
    hit_dropped = bool(missing)
    if missing:
        print(f"   缺失的核心概念: {missing}")

    print("   [判定] 出现 >=2 个 'AND all:'（AND 坍缩）或核心概念缺失（首短语截断丢限定词）")
    hit = hit_and or hit_dropped
    _verdict(hit)
    return hit


# ---------------------------------------------------------------------------
# 3. OpenReview: results are truncated to `limit` first, then filtered by the
#    requested date window locally, so most slots are discarded.
# ---------------------------------------------------------------------------
def _openreview_note(index: int, in_window: bool) -> dict:
    published = datetime(2025, 6, 1, tzinfo=UTC) if in_window else datetime(2022, 1, 1, tzinfo=UTC)
    return {
        "id": f"note-{index}",
        "pdate": int(published.timestamp() * 1000),
        "content": {
            "title": {"value": f"LLM agents paper {index}"},
            "abstract": {"value": "abstract"},
        },
    }


def check_openreview_truncation() -> bool:
    _title("问题3：OpenReview 先按 limit 截断，再在本地过滤日期")

    # Every 4th note falls inside the requested window; the rest are older.
    pool = [_openreview_note(i, in_window=(i % 4 == 0)) for i in range(40)]
    expected_in_window = sum(1 for i in range(40) if i % 4 == 0)

    captured: dict = {}

    async def run(limit: int) -> list[Paper]:
        plan = SearchPlan(
            intent_summary="llm agents",
            queries=[
                SearchQuery(
                    source=PaperSource.OPENREVIEW,
                    query="large language model agents",
                    limit=limit,
                    filters={"date_from": "2024-10-27", "date_to": "2026-10-27"},
                )
            ],
        )
        retriever = OpenReviewRetriever()

        def fake_fetch(raw, cap, filters=None):
            captured["cap"] = cap
            return {"notes": pool[:cap]}

        retriever._fetch = fake_fetch
        return await retriever.search(plan)

    small = asyncio.run(run(7))
    fetched_cap = captured["cap"]
    in_window_fetched = sum(1 for i in range(fetched_cap) if i % 4 == 0)
    print(f"   请求窗口内可用记录: {expected_in_window} 条（模拟数据池 40 条）")
    print(f"   limit=7  实际 fetch cap : {fetched_cap}（过采样应 > 7）")
    print(
        f"   limit=7  最终返回       : {len(small)} 条（fetch 范围内窗口内记录 {in_window_fetched} 条）"
    )
    # 修复后应过采样（cap > limit），否则先按 limit 截断再过滤会把窗口内记录丢掉。
    hit = fetched_cap <= 7 or len(small) < min(in_window_fetched, 7)
    print("   [判定] fetch cap 未过采样（<=limit）=> 先截断再过滤丢弃窗口内记录")
    _verdict(hit)
    return hit


# ---------------------------------------------------------------------------
# 4. PaperMerger: an arXiv-only record and a DOI-only record of the same paper
#    are never merged, because any unmatched identifier blocks title matching.
# ---------------------------------------------------------------------------
def check_merger_arxiv_doi() -> bool:
    _title("问题4：跨源去重拒绝合并 arXiv 版与 DOI 版")

    title = "Task Offloading in Edge Computing: A Survey"
    author = Author(name="Alice Zhang")

    doi_paper = Paper(title=title, authors=[author], year=2024, doi="10.1109/TMC.2024.1234567")
    arxiv_paper = Paper(title=title, authors=[author], year=2024, arxiv_id="2401.01234v1")
    merged = asyncio.run(PaperMerger().merge_and_deduplicate([doi_paper, arxiv_paper]))

    control_a = Paper(title=title, authors=[author], year=2024)
    control_b = Paper(title=title, authors=[author], year=2024)
    control = asyncio.run(PaperMerger().merge_and_deduplicate([control_a, control_b]))

    print(f"   DOI 版 + arXiv 版   : {len(merged)} 条（期望 1 条）")
    print(f"   对照组（都无标识符）: {len(control)} 条（期望 1 条）")
    hit = len(merged) != 1
    print("   [判定] 只有带不同标识符时无法合并 => 标题/作者匹配被标识符边界阻断")
    _verdict(hit)
    return hit


# ---------------------------------------------------------------------------
# Optional: count real duplicates left in the newest saved pipeline run.
# ---------------------------------------------------------------------------
def report_saved_run_duplicates() -> None:
    _title("附：最近一次运行结果里的同标题重复")
    files = sorted((ROOT / ".playwright-cli").glob("pipeline-result-*.json"))
    if not files:
        print("   未找到 .playwright-cli/pipeline-result-*.json，跳过")
        return
    latest = files[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    papers = data.get("metadata_papers") or data.get("papers") or []
    groups: dict[str, list[dict]] = {}
    for item in papers:
        groups.setdefault(normalize_title(item.get("title") or ""), []).append(item)
    duplicates = {key: items for key, items in groups.items() if len(items) > 1}
    print(f"   文件: {latest.name}")
    print(f"   元数据论文 {len(papers)} 条，同标题分组 {len(duplicates)} 组")
    for key, items in list(duplicates.items())[:5]:
        ids = [(item.get("doi"), item.get("arxiv_id")) for item in items]
        print(f"   - {key[:60]!r} -> {ids}")
    if not duplicates:
        print("   （本次样本未发现重复；样本较小，不能据此认为去重没有问题）")


# ---------------------------------------------------------------------------
# Optional live probes.
# ---------------------------------------------------------------------------
def live_s2_bulk() -> None:
    _title("Live：Semantic Scholar /bulk 实际返回")
    import requests

    params = {
        "query": '"large language model agents"',
        "fields": "paperId,title,abstract,year,publicationDate",
        "publicationDateOrYear": "2024-10-27:2026-10-27",
        "sort": "publicationDate:desc",
    }
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            params=params,
            timeout=30,
        )
        body = response.json() if response.status_code == 200 else None
        count = len(body.get("data", [])) if body else 0
        print(
            f"   status={response.status_code} 返回条数={count} 响应体={len(response.content)} bytes"
        )
        if response.status_code != 200:
            print(f"   错误响应: {response.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   请求失败: {type(exc).__name__}: {exc}")


def live_arxiv_recall() -> None:
    _title("Live：arXiv 三短语 AND vs 单短语 的 totalResults")
    import requests

    pairs = [
        ('all:"task offloading" AND all:"edge computing" AND all:5g', "3 短语 AND"),
        ('all:"task offloading"', "单短语"),
    ]
    for expression, label in pairs:
        try:
            response = requests.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": expression, "max_results": 1},
                timeout=30,
            )
            match = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", response.text)
            print(
                f"   {label:<10} status={response.status_code} totalResults={match.group(1) if match else '?'}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   {label:<10} 请求失败: {type(exc).__name__}: {exc}")
        time.sleep(3.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also probe the real APIs")
    args = parser.parse_args()

    hits = [
        check_s2_bulk(),
        check_arxiv_and(),
        check_openreview_truncation(),
        check_merger_arxiv_doi(),
    ]
    report_saved_run_duplicates()

    if args.live:
        live_s2_bulk()
        live_arxiv_recall()

    print("\n== 汇总 ==")
    labels = ["S2 /bulk", "arXiv AND", "OpenReview 截断", "arXiv/DOI 去重"]
    for label, hit in zip(labels, hits, strict=True):
        print(f"   {label:<18} {'命中' if hit else '未命中'}")
    return 1 if any(hits) else 0


if __name__ == "__main__":
    sys.exit(main())
