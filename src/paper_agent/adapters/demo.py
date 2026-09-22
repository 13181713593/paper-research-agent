"""Offline demo components.

These implementations prove the interfaces and UI wiring. Work groups should replace
them one port at a time; they deliberately do not pretend to access real databases.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from paper_agent.application.pipeline import PipelineDependencies
from paper_agent.domain.models import (
    AccessStatus,
    Author,
    EvidenceRef,
    FullTextAsset,
    Paper,
    PaperSource,
    PaperSummary,
    ResearchIdea,
    ResearchReport,
    ScoreBreakdown,
    SearchPlan,
    SearchQuery,
    SourceRecord,
    Venue,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).strip()


class DemoQueryPlanner:
    async def plan(self, idea: ResearchIdea, limit_per_source: int) -> SearchPlan:
        query = " ".join([idea.text, *idea.required_terms])
        return SearchPlan(
            intent_summary=idea.text,
            queries=[
                SearchQuery(source=PaperSource.ARXIV, query=query, limit=limit_per_source),
                SearchQuery(source=PaperSource.OPENALEX, query=query, limit=limit_per_source),
            ],
            inclusion_criteria=["Directly addresses the research idea", "Contains usable metadata"],
            exclusion_criteria=idea.excluded_terms,
        )


class DemoRetriever:
    def __init__(self, source: PaperSource):
        self.source = source
        self.source_name = source.value

    async def search(self, plan: SearchPlan) -> list[Paper]:
        idea = plan.intent_summary
        shared = Paper(
            title=f"A Unified Study of {idea[:55]}",
            authors=[Author(name="Alex Chen", affiliations=["Example University"])],
            institutions=["Example University"],
            publication_date=date(2026, 5, 1),
            venue=Venue(name="Demo Conference", venue_type="conference"),
            abstract=f"We study {idea} with a reproducible benchmark and evidence-aware ranking.",
            doi="10.0000/demo.shared",
            is_open_access=True,
            fulltext=FullTextAsset(
                pdf_url="https://example.org/papers/shared.pdf",
                access_status=AccessStatus.OPEN_ACCESS,
            ),
            source_records=[SourceRecord(source=self.source, source_id=f"{self.source}-shared")],
        )
        unique = Paper(
            title=f"{self.source.value.title()} Evidence for {idea[:50]}",
            authors=[Author(name="Taylor Li")],
            publication_date=date(2025, 11, 20),
            venue=Venue(name="Demo Journal", venue_type="journal"),
            abstract=f"A source-specific investigation of {idea}, including experiments and limits.",
            arxiv_id=f"2601.{1 if self.source == PaperSource.ARXIV else 2:05d}",
            is_open_access=True,
            fulltext=FullTextAsset(
                pdf_url=f"https://example.org/papers/{self.source.value}.pdf",
                access_status=AccessStatus.OPEN_ACCESS,
            ),
            source_records=[SourceRecord(source=self.source, source_id=f"{self.source}-unique")],
        )
        return [shared, unique]


class DemoMerger:
    async def merge_and_deduplicate(self, papers: Sequence[Paper]) -> list[Paper]:
        merged: dict[str, Paper] = {}
        for paper in papers:
            key = paper.doi or paper.arxiv_id or _normalize(paper.title)
            if key not in merged:
                paper.normalized_title = _normalize(paper.title)
                merged[key] = paper
                continue
            canonical = merged[key]
            canonical.source_records.extend(paper.source_records)
            canonical.aliases.append(paper.title)
            if not canonical.abstract and paper.abstract:
                canonical.abstract = paper.abstract
        return list(merged.values())


class DemoRanker:
    def __init__(self, final: bool = False):
        self.final = final

    async def rank(self, idea: ResearchIdea, papers: Sequence[Paper], limit: int) -> list[Paper]:
        for paper in papers:
            similarity = SequenceMatcher(
                None, _normalize(idea.text), _normalize(f"{paper.title} {paper.abstract or ''}")
            ).ratio()
            recency = min(1.0, max(0.0, ((paper.year or 2000) - 2020) / 6))
            evidence = paper.fulltext.parse_quality or (0.4 if paper.abstract else 0.0)
            final = 0.65 * similarity + 0.2 * recency + 0.15 * evidence
            paper.scores = ScoreBreakdown(
                semantic=similarity,
                recency=recency,
                evidence_coverage=evidence if self.final else None,
                final=min(1.0, final),
                formula_version="demo-v1-final" if self.final else "demo-v1-pre",
                explanation="Demo score; replace with an evaluated academic reranker.",
            )
        return sorted(papers, key=lambda p: p.scores.final or 0, reverse=True)[:limit]


class DemoFullTextProcessor:
    async def process(self, paper: Paper) -> Paper:
        if paper.fulltext.access_status != AccessStatus.OPEN_ACCESS:
            raise PermissionError("No legal open-access full text")
        paper.fulltext.markdown_path = str(Path("data/markdown") / f"{paper.paper_id}.md")
        paper.fulltext.parser_name = "demo-parser"
        paper.fulltext.parser_version = "0.1"
        paper.fulltext.parse_quality = 0.9
        return paper


class DemoVerifier:
    async def verify(self, idea: ResearchIdea, paper: Paper) -> Paper:
        paper.scores.evidence_coverage = 0.9 if paper.fulltext.markdown_path else 0.4
        return paper


class DemoSummarizer:
    async def summarize(self, idea: ResearchIdea, paper: Paper) -> PaperSummary:
        evidence = [EvidenceRef(paper_id=paper.paper_id, section="Abstract")]
        return PaperSummary(
            paper_id=paper.paper_id,
            core_problem=f"How to address: {idea.text}",
            method="Uses a benchmarked, evidence-aware pipeline (demo text).",
            experiments="Reports comparative experiments (demo text).",
            contributions=["A reusable pipeline", "Explicit evidence tracking"],
            limitations=["This is an offline demo record, not a real-paper conclusion"],
            relevance_to_idea="Direct lexical and semantic overlap with the stated idea.",
            evidence=evidence,
            generated_from_fulltext=bool(paper.fulltext.markdown_path),
        )


class DemoReportWriter:
    async def write(
        self, idea: ResearchIdea, papers: Sequence[Paper], summaries: Sequence[PaperSummary]
    ) -> ResearchReport:
        rows = "\n".join(
            f"{i}. **{paper.title}** — score {paper.scores.final or 0:.3f}"
            for i, paper in enumerate(papers, 1)
        )
        markdown = f"# Research report\n\n## Idea\n\n{idea.text}\n\n## Top papers\n\n{rows}\n"
        return ResearchReport(
            title="Research reference report",
            executive_summary=f"Selected {len(papers)} papers for the supplied idea.",
            markdown=markdown,
            paper_ids=[p.paper_id for p in papers],
            future_directions=["Replace demo adapters with evaluated production components"],
        )


def build_demo_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        query_planner=DemoQueryPlanner(),
        retrievers=[DemoRetriever(PaperSource.ARXIV), DemoRetriever(PaperSource.OPENALEX)],
        merger=DemoMerger(),
        pre_ranker=DemoRanker(final=False),
        fulltext_processor=DemoFullTextProcessor(),
        verifier=DemoVerifier(),
        final_ranker=DemoRanker(final=True),
        summarizer=DemoSummarizer(),
        report_writer=DemoReportWriter(),
    )

