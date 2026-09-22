"""Shared domain contracts.

Rules:
- Unknown external metadata is ``None``; never invent an empty string.
- Lists use empty defaults because "known empty" is meaningful and serializable.
- Every derived claim can point to evidence through ``EvidenceRef``.
- Raw provider payloads are isolated in ``source_records`` to keep the canonical model stable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class PaperSource(StrEnum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    OPENALEX = "openalex"
    CORE = "core"
    CNKI = "cnki"
    OTHER = "other"


class AccessStatus(StrEnum):
    UNKNOWN = "unknown"
    OPEN_ACCESS = "open_access"
    CLOSED = "closed"
    UNAVAILABLE = "unavailable"


class PublicationStatus(StrEnum):
    """Publication lifecycle; do not infer peer review from a venue string."""

    UNKNOWN = "unknown"
    PREPRINT = "preprint"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    RETRACTED = "retracted"


class CodeAvailability(StrEnum):
    """Whether an implementation is available under an identifiable license."""

    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"
    REPOSITORY_FOUND = "repository_found"
    OPEN_SOURCE = "open_source"
    CLOSED_SOURCE = "closed_source"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage(StrEnum):
    CREATED = "created"
    QUERY_PLANNED = "query_planned"
    RETRIEVED = "retrieved"
    DEDUPED = "deduped"
    PRE_RANKED = "pre_ranked"
    FULLTEXT_PROCESSED = "fulltext_processed"
    VERIFIED = "verified"
    FINAL_RANKED = "final_ranked"
    SUMMARIZED = "summarized"
    REPORTED = "reported"
    FAILED = "failed"


class Author(StrictModel):
    name: str = Field(min_length=1)
    normalized_name: str | None = None
    orcid: str | None = None
    affiliations: list[str] = Field(default_factory=list)
    corresponding: bool | None = None


class Venue(StrictModel):
    name: str | None = None
    short_name: str | None = None
    venue_type: str | None = Field(default=None, description="journal/conference/workshop/preprint")
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None


class SourceRecord(StrictModel):
    source: PaperSource
    source_id: str
    source_url: HttpUrl | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class FullTextAsset(StrictModel):
    pdf_url: HttpUrl | None = None
    landing_page_url: HttpUrl | None = None
    local_pdf_path: str | None = None
    markdown_path: str | None = None
    access_status: AccessStatus = AccessStatus.UNKNOWN
    license: str | None = None
    version: str | None = None
    sha256: str | None = None
    downloaded_at: datetime | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    parse_quality: Annotated[float | None, Field(ge=0, le=1)] = None
    parse_error: str | None = None


class CodeAsset(StrictModel):
    """Code availability is separate from paper open-access status."""

    status: CodeAvailability = CodeAvailability.UNKNOWN
    repository_url: HttpUrl | None = None
    license: str | None = None
    last_verified_at: datetime | None = None


class ScoreBreakdown(StrictModel):
    lexical: Annotated[float | None, Field(ge=0, le=1)] = None
    semantic: Annotated[float | None, Field(ge=0, le=1)] = None
    reranker: Annotated[float | None, Field(ge=0, le=1)] = None
    recency: Annotated[float | None, Field(ge=0, le=1)] = None
    venue_quality: Annotated[float | None, Field(ge=0, le=1)] = None
    intent_match: Annotated[float | None, Field(ge=0, le=1)] = None
    evidence_coverage: Annotated[float | None, Field(ge=0, le=1)] = None
    final: Annotated[float | None, Field(ge=0, le=1)] = None
    formula_version: str | None = None
    explanation: str | None = None


class Paper(StrictModel):
    """Canonical paper entity shared by all groups.

    Required fields are intentionally minimal because public APIs often return
    incomplete records. Unknown scalar metadata is ``None``; list fields are
    empty only when no values were supplied. Provenance always belongs in
    ``source_records`` so merged values remain auditable.
    """

    paper_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    normalized_title: str | None = None
    authors: list[Author] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    year: int | None = Field(default=None, ge=1600, le=2200)
    publication_status: PublicationStatus = PublicationStatus.UNKNOWN
    venue: Venue | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    comments: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    pmid: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    bibtex: str | None = None
    language: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    is_retracted: bool | None = None
    is_open_access: bool | None = None
    open_access_url: HttpUrl | None = None
    landing_page_url: HttpUrl | None = None
    project_page_url: HttpUrl | None = None
    dataset_urls: list[HttpUrl] = Field(default_factory=list)
    code: CodeAsset = Field(default_factory=CodeAsset)
    fulltext: FullTextAsset = Field(default_factory=FullTextAsset)
    source_records: list[SourceRecord] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    duplicate_of: UUID | None = None
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    metadata_conflicts: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def align_year(self) -> Paper:
        if self.publication_date and self.year and self.publication_date.year != self.year:
            raise ValueError("year must match publication_date.year")
        if self.publication_date and self.year is None:
            self.year = self.publication_date.year
        return self


class ResearchIdea(StrictModel):
    text: str = Field(min_length=10)
    preferred_languages: list[str] = Field(default_factory=lambda: ["en", "zh"])
    date_from: date | None = None
    date_to: date | None = None
    required_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    preferred_venues: list[str] = Field(default_factory=list)
    ranking_focus: list[str] = Field(default_factory=list)


class SearchQuery(StrictModel):
    source: PaperSource
    query: str
    limit: int = Field(default=30, ge=1, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)


class SearchPlan(StrictModel):
    intent_summary: str
    queries: list[SearchQuery] = Field(min_length=1)
    synonyms: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class EvidenceRef(StrictModel):
    paper_id: UUID
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    quote: str | None = Field(default=None, max_length=500)
    markdown_anchor: str | None = None


class PaperSummary(StrictModel):
    paper_id: UUID
    core_problem: str
    method: str
    experiments: str
    contributions: list[str]
    limitations: list[str]
    relevance_to_idea: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    generated_from_fulltext: bool


class ResearchReport(StrictModel):
    title: str
    executive_summary: str
    markdown: str
    paper_ids: list[UUID]
    future_directions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class StageEvent(StrictModel):
    stage: PipelineStage
    status: ProcessingStatus
    started_at: datetime
    finished_at: datetime | None = None
    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    message: str | None = None
    error_code: str | None = None


class PipelineRequest(StrictModel):
    idea: ResearchIdea
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_limit_per_source: int = Field(default=30, ge=1, le=200)
    pre_rank_pool_size: int = Field(default=20, ge=1, le=100)
    require_fulltext: bool = False
    allow_abstract_fallback: bool = True

    @model_validator(mode="after")
    def pool_covers_top_k(self) -> PipelineRequest:
        if self.pre_rank_pool_size < self.top_k:
            raise ValueError("pre_rank_pool_size must be >= top_k")
        return self


class PipelineResult(StrictModel):
    run_id: UUID = Field(default_factory=uuid4)
    status: ProcessingStatus
    stage: PipelineStage
    request: PipelineRequest
    papers: list[Paper] = Field(default_factory=list)
    summaries: list[PaperSummary] = Field(default_factory=list)
    report: ResearchReport | None = None
    events: list[StageEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_duration_ms(self) -> int:
        """Wall-time approximation from completed stage events."""
        return sum(event.duration_ms or 0 for event in self.events)


class PipelineContract(StrictModel):
    """Machine-readable description used by MCP hosts and the UI."""

    name: str = "paper-research-agent"
    version: str
    stages: list[PipelineStage]
    minimum_sources: int = 2
    supports_abstract_fallback: bool = True
    notes: list[str] = Field(default_factory=list)
