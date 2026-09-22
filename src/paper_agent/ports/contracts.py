"""Interfaces owned by the overall architecture; each work group supplies implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from paper_agent.domain.models import (
    Paper,
    PaperSummary,
    ResearchIdea,
    ResearchReport,
    SearchPlan,
)

ProgressCallback = Callable[[str, int, int, str], Awaitable[None]]


class QueryPlanner(Protocol):
    async def plan(self, idea: ResearchIdea, limit_per_source: int) -> SearchPlan:
        """Group 1: rewrite an idea into source-specific queries.

        Input: normalized research idea and per-provider upper bound.
        Output: ``SearchPlan`` with one or more provider-specific queries plus
        explicit inclusion/exclusion criteria. Never perform network I/O here.
        """


class BibTexMapper(Protocol):
    async def to_paper(
        self,
        bibtex: str,
        *,
        source: str,
        supplemental_metadata: dict[str, Any] | None = None,
    ) -> Paper:
        """Group 1: convert a BibTeX entry and provider metadata to ``Paper``.

        Input: one BibTeX entry, source label, optional provider payload.
        Output: validated canonical ``Paper``. Missing fields remain ``None``;
        provider payload/provenance must be retained in ``source_records``.
        """


class PaperRetriever(Protocol):
    source_name: str

    async def search(self, plan: SearchPlan) -> list[Paper]:
        """Group 1: return metadata only for this retriever's query.

        Input: full ``SearchPlan``; implementation selects queries matching its
        source. Output: independently validated ``Paper`` records. It must not
        download PDFs, perform final Top-k selection, or silently drop provider
        errors. Rate limits should raise a typed provider error.
        """


class PaperMerger(Protocol):
    async def merge_and_deduplicate(self, papers: Sequence[Paper]) -> list[Paper]:
        """Group 1: merge DOI/arXiv exact matches, then cautious fuzzy matches.

        Input: records from every provider. Output: canonical records preserving
        all source records, aliases and metadata conflicts. Conference and journal
        extensions must not be merged merely because their titles are similar.
        """


class PreRanker(Protocol):
    async def rank(self, idea: ResearchIdea, papers: Sequence[Paper], limit: int) -> list[Paper]:
        """Group 2: cheap, high-recall metadata rank for full-text pool selection.

        Input: idea, deduplicated papers, pool size N. Output: at most N papers,
        ordered, with normalized score breakdown. This is explicitly not Top-k.
        """


class RankingIntentRewriter(Protocol):
    async def rewrite(self, idea: ResearchIdea) -> ResearchIdea:
        """Group 2: extract ranking emphasis without changing the user's intent.

        Output may populate required/excluded terms and ``ranking_focus``. The
        original ``text`` must be retained for traceability.
        """


class OpenAccessResolver(Protocol):
    async def resolve(self, paper: Paper) -> Paper:
        """Group 3: locate a lawful OA copy and update URL/access metadata.

        No paywall bypassing. ``is_open_access`` is provider metadata, while
        ``fulltext.access_status`` is the resolver's verified observation.
        """


class PdfDownloader(Protocol):
    async def download(self, paper: Paper, destination_dir: Path) -> Path:
        """Group 3: download and validate one PDF, returning a local path.

        Enforce timeout, maximum bytes, redirect policy, MIME/PDF magic check,
        safe filename, checksum cache and retry limits. Raise typed errors.
        """


class MarkdownParser(Protocol):
    async def parse(self, pdf_path: Path, destination_dir: Path) -> tuple[Path, float]:
        """Group 3: convert a local PDF to Markdown.

        Output is ``(markdown_path, quality_score)``. The score is in [0, 1]
        and should reflect text coverage, heading/table integrity and OCR quality.
        """


class FullTextProcessor(Protocol):
    async def process(self, paper: Paper) -> Paper:
        """Group 3 facade combining OA resolution, download and Markdown parsing.

        Input/output use the same paper identity. The implementation updates only
        ``fulltext`` fields and must remain safe for bounded concurrent calls.
        """


class EvidenceVerifier(Protocol):
    async def verify(self, idea: ResearchIdea, paper: Paper) -> Paper:
        """Group 2: confirm inclusion criteria using abstract/full-text evidence."""


class FinalRanker(Protocol):
    async def rank(self, idea: ResearchIdea, papers: Sequence[Paper], limit: int) -> list[Paper]:
        """Group 2: expensive evidence-aware rerank returning final Top-k.

        Input papers have completed evidence verification. Output must carry a
        score formula version, component scores and a human-readable rationale.
        """


class PaperSummarizer(Protocol):
    async def summarize(self, idea: ResearchIdea, paper: Paper) -> PaperSummary:
        """Group 3: create an evidence-linked structured summary.

        Abstract-only fallback must set ``generated_from_fulltext=False`` and
        must not invent experimental settings absent from available evidence.
        """


class ReportWriter(Protocol):
    async def write(
        self, idea: ResearchIdea, papers: Sequence[Paper], summaries: Sequence[PaperSummary]
    ) -> ResearchReport:
        """Group 3: synthesize a Markdown report from validated summaries.

        Future directions are model recommendations, not paper claims, and
        should be labeled accordingly. Every factual claim should be traceable.
        """
