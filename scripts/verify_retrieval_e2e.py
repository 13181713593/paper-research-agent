"""Retrieval-group end-to-end verification (planner -> 4 sources -> merger).

Real network round trip over the production retrieval segment assembled by
``paper_agent.bootstrap.build_retrieval_dependencies``:

1. Real ``HeuristicQueryPlanner`` turns a research idea into one
   space-separated keyword query per source (``SearchPlan``).
2. All four live retrievers (arXiv, Crossref, OpenAlex, Semantic Scholar)
   each consume only their own query and return canonical ``Paper`` objects.
3. ``PaperMerger`` merges and deduplicates; every ``SourceRecord`` from
   every source must survive the merge.

A single failed source degrades to a warning (matching pipeline semantics);
failure of ALL sources fails the run. Usage::

    python scripts/verify_retrieval_e2e.py [--idea "..." ] [--limit 3]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from paper_agent.bootstrap import build_retrieval_dependencies
from paper_agent.domain.models import Paper, ResearchIdea

DEFAULT_IDEA = "用强化学习提升小语言模型的数学推理能力"


def _source_counts(papers: list[Paper]) -> Counter[str]:
    return Counter(r.source.value for p in papers for r in p.source_records)


async def _run(idea: ResearchIdea, limit: int) -> int:
    deps = build_retrieval_dependencies()

    print("== 1. Query plan (real HeuristicQueryPlanner, no network) ==")
    plan = await deps.query_planner.plan(idea, limit)
    print(f"   intent: {idea.text}")
    for query in plan.queries:
        print(f"   {query.source.value:<17} limit={query.limit:<3} q={query.query!r}")

    print("\n== 2. Live retrieval (4 sources, tolerant of per-source failure) ==")
    results = await asyncio.gather(
        *(retriever.search(plan) for retriever in deps.retrievers),
        return_exceptions=True,
    )
    papers: list[Paper] = []
    failures: list[str] = []
    for retriever, result in zip(deps.retrievers, results, strict=True):
        if isinstance(result, BaseException):
            failures.append(f"{retriever.source_name}: {type(result).__name__}: {result}")
            print(f"   [WARN] {retriever.source_name:<17} failed -> {type(result).__name__}")
            continue
        with_abstract = sum(1 for p in result if p.abstract)
        print(
            f"   [OK]   {retriever.source_name:<17} {len(result)} papers, "
            f"{with_abstract} with abstract"
        )
        for p in result[:2]:
            print(f"         e.g. {p.title[:80]}")
        papers.extend(result)

    if not papers:
        print("\n[FAIL] All sources failed or returned nothing.")
        return 1

    before = _source_counts(papers)
    print("\n== 3. Merge & deduplicate (real PaperMerger) ==")
    print(f"   raw papers: {len(papers)}, source records: {sum(before.values())}")
    merged = await deps.merger.merge_and_deduplicate(papers)
    after = _source_counts(merged)
    print(f"   merged papers: {len(merged)}, source records: {sum(after.values())}")
    multi = [p for p in merged if len(p.source_records) > 1]
    print(f"   papers carrying >=2 sources (cross-source dedup hits): {len(multi)}")
    for p in multi[:3]:
        print(f"         e.g. {p.title[:70]} <- {[r.source.value for r in p.source_records]}")

    print("\n== 4. Contract checks ==")
    ok = True
    if sum(before.values()) != sum(after.values()):
        print("   [FAIL] source_records lost during merge")
        ok = False
    else:
        print(f"   [OK] all {sum(after.values())} source_records preserved")
    with_abstract = sum(1 for p in merged if p.abstract)
    print(
        f"   [OK] {with_abstract}/{len(merged)} merged papers carry an abstract "
        f"(missing ones stay None, never fabricated)"
    )
    invented = [p for p in merged if p.abstract == "unknown" or p.abstract == ""]
    if invented:
        print("   [FAIL] fabricated placeholder abstracts found")
        ok = False
    else:
        print("   [OK] no placeholder/fabricated abstract values")

    status = "PASS" if ok else "FAIL"
    print(f"\nRetrieval E2E: {status} (source failures tolerated: {len(failures)})")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default=DEFAULT_IDEA, help="research idea text")
    parser.add_argument("--limit", type=int, default=3, help="limit per source")
    args = parser.parse_args()
    idea = ResearchIdea(
        text=args.idea,
        required_terms=[],
        excluded_terms=[],
    )
    return asyncio.run(_run(idea, args.limit))


if __name__ == "__main__":
    sys.exit(main())
