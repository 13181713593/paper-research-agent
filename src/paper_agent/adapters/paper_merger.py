"""Conservative, provenance-preserving paper deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from paper_agent.domain.models import (
    Author,
    FullTextAsset,
    Paper,
    PublicationStatus,
    SourceRecord,
    Venue,
)

from .retrieval_common import normalize_arxiv, normalize_doi, normalize_title


class PaperMerger:
    """Merge records without mutating provider-owned input objects."""

    async def merge_and_deduplicate(self, papers: Sequence[Paper]) -> list[Paper]:
        groups: list[Paper] = []
        for original in papers:
            candidate = deepcopy(original)
            self._normalize(candidate)
            match = next(
                (existing for existing in groups if self._can_merge(existing, candidate)), None
            )
            if match is None:
                groups.append(candidate)
            else:
                self._merge_into(match, candidate)
                # A newly supplied DOI/arXiv pair may connect two groups that
                # previously had no shared identifier. Recheck those groups
                # until no strong match remains; keep the DOI-conflict guard.
                index = 0
                while (match.doi or match.arxiv_id) and index < len(groups):
                    existing = groups[index]
                    if existing is not match and self._can_merge(match, existing):
                        self._merge_into(match, existing)
                        groups.pop(index)
                        index = 0
                    else:
                        index += 1
        return groups

    @staticmethod
    def _normalize(paper: Paper) -> None:
        paper.doi = normalize_doi(paper.doi)
        paper.arxiv_id = normalize_arxiv(paper.arxiv_id)
        paper.normalized_title = normalize_title(paper.title)

    @classmethod
    def _can_merge(cls, left: Paper, right: Paper) -> bool:
        left_doi, right_doi = normalize_doi(left.doi), normalize_doi(right.doi)
        if left_doi and right_doi:
            return left_doi == right_doi
        left_arxiv = normalize_arxiv(left.arxiv_id, strip_version=True)
        right_arxiv = normalize_arxiv(right.arxiv_id, strip_version=True)
        if left_arxiv and right_arxiv:
            return left_arxiv == right_arxiv
        # An unmatched DOI is a strong boundary for title-only matching.
        if left_doi or right_doi:
            return False
        if left_arxiv or right_arxiv:
            return False
        for field in ("pmid", "openalex_id", "semantic_scholar_id"):
            left_id, right_id = getattr(left, field), getattr(right, field)
            if left_id and right_id and left_id != right_id:
                return False
        if normalize_title(left.title) != normalize_title(right.title):
            return False
        if cls._venue_boundary(left.venue, right.venue):
            return False
        if left.year and right.year and abs(left.year - right.year) > 1:
            return False
        left_authors = {normalize_title(author.name) for author in left.authors}
        right_authors = {normalize_title(author.name) for author in right.authors}
        return not (left_authors and right_authors and not (left_authors & right_authors))

    @staticmethod
    def _venue_boundary(left: Venue | None, right: Venue | None) -> bool:
        if not left or not right:
            return False
        left_type, right_type = (
            (left.venue_type or "").casefold(),
            (right.venue_type or "").casefold(),
        )
        return bool(left_type and right_type and left_type != right_type)

    @classmethod
    def _merge_into(cls, target: Paper, incoming: Paper) -> None:
        if target.title != incoming.title and normalize_title(target.title) == normalize_title(
            incoming.title
        ):
            target.aliases = cls._unique(target.aliases + [incoming.title])
        elif target.title != incoming.title:
            cls._add_conflict(
                target, f"title: conflicting values retained first ({incoming.title!r})"
            )
        target.aliases = cls._unique(target.aliases + incoming.aliases)
        target.authors = cls._merge_authors(target.authors, incoming.authors)
        target.institutions = cls._unique(target.institutions + incoming.institutions)
        target.keywords = cls._unique(target.keywords + incoming.keywords)
        target.source_records = cls._merge_sources(target.source_records, incoming.source_records)
        target.metadata_conflicts = cls._unique(
            target.metadata_conflicts + incoming.metadata_conflicts
        )

        for field in (
            "doi",
            "arxiv_id",
            "pmid",
            "openalex_id",
            "semantic_scholar_id",
            "bibtex",
            "language",
            "comments",
            "landing_page_url",
            "open_access_url",
            "project_page_url",
        ):
            cls._choose_scalar(target, incoming, field)
        if target.publication_status == PublicationStatus.UNKNOWN:
            target.publication_status = incoming.publication_status
        elif (
            incoming.publication_status != PublicationStatus.UNKNOWN
            and target.publication_status != incoming.publication_status
        ):
            cls._add_conflict(target, "publication_status: conflicting values retained first")
        cls._merge_publication_time(target, incoming)
        cls._choose_scalar(target, incoming, "abstract", prefer_longer=True)
        if target.venue is None and incoming.venue is not None:
            target.venue = deepcopy(incoming.venue)
        elif target.venue is not None and incoming.venue is not None:
            for field in Venue.model_fields:
                cls._merge_asset_field(
                    target, target.venue, incoming.venue, field, f"venue.{field}"
                )
        target.citation_count = cls._max_optional(target.citation_count, incoming.citation_count)
        if incoming.is_open_access is not None:
            target.is_open_access = (
                target.is_open_access
                if target.is_open_access is not None
                else incoming.is_open_access
            )
        if incoming.is_retracted is not None:
            target.is_retracted = (
                target.is_retracted if target.is_retracted is not None else incoming.is_retracted
            )
        target.dataset_urls = cls._unique(target.dataset_urls + incoming.dataset_urls)
        cls._merge_code(target, incoming)
        cls._merge_fulltext(target, incoming)
        target.updated_at = max(target.updated_at, incoming.updated_at)

    @classmethod
    def _merge_code(cls, target: Paper, incoming: Paper) -> None:
        """Combine code metadata without replacing fields from another group."""
        current, candidate = target.code, incoming.code
        if current.status.value == "unknown" and candidate.status.value != "unknown":
            current.status = candidate.status
        elif (
            current.status.value != "unknown"
            and candidate.status.value != "unknown"
            and current.status != candidate.status
        ):
            cls._add_conflict(target, "code.status: conflicting values retained first")
        for field in ("repository_url", "license", "last_verified_at"):
            cls._merge_asset_field(target, current, candidate, field, f"code.{field}")

    @classmethod
    def _merge_fulltext(cls, target: Paper, incoming: Paper) -> None:
        """Combine full-text assets field by field to preserve parser output."""
        current, candidate = target.fulltext, incoming.fulltext
        if current.access_status.value == "unknown" and candidate.access_status.value != "unknown":
            current.access_status = candidate.access_status
        elif (
            current.access_status.value != "unknown"
            and candidate.access_status.value != "unknown"
            and current.access_status != candidate.access_status
        ):
            cls._add_conflict(target, "fulltext.access_status: conflicting values retained first")
        for field in FullTextAsset.model_fields:
            if field == "access_status":
                continue
            if field in {"candidate_urls", "attempts"}:
                setattr(
                    current, field, cls._unique(getattr(current, field) + getattr(candidate, field))
                )
                continue
            cls._merge_asset_field(target, current, candidate, field, f"fulltext.{field}")

    @classmethod
    def _merge_asset_field(cls, target: Paper, current, candidate, field: str, label: str) -> None:
        left, right = getattr(current, field), getattr(candidate, field)
        if left is None and right is not None:
            setattr(current, field, deepcopy(right))
        elif left is not None and right is not None and left != right:
            cls._add_conflict(target, f"{label}: conflicting values retained first")

    @staticmethod
    def _choose_scalar(
        target: Paper, incoming: Paper, field: str, prefer_longer: bool = False
    ) -> None:
        current, candidate = getattr(target, field), getattr(incoming, field)
        if current is None and candidate is not None:
            setattr(target, field, deepcopy(candidate))
        elif current is not None and candidate is not None and current != candidate:
            if (
                prefer_longer
                and isinstance(current, str)
                and isinstance(candidate, str)
                and len(candidate) > len(current)
            ):
                setattr(target, field, candidate)
            else:
                PaperMerger._add_conflict(target, f"{field}: conflicting values retained first")

    @staticmethod
    def _merge_publication_time(target: Paper, incoming: Paper) -> None:
        incoming_date, incoming_year = incoming.publication_date, incoming.year
        if target.publication_date is not None:
            if incoming_date and target.publication_date != incoming_date:
                PaperMerger._add_conflict(
                    target, "publication_date: conflicting values retained first"
                )
            if incoming_year and incoming_year != target.publication_date.year:
                PaperMerger._add_conflict(target, "year: conflicts with retained publication_date")
            return
        if incoming_date is not None:
            if target.year is not None and target.year != incoming_date.year:
                PaperMerger._add_conflict(target, "publication_date: conflicts with retained year")
            else:
                target.publication_date = incoming_date
                target.year = incoming_date.year
                return
        if target.year is None and incoming_year is not None:
            target.year = incoming_year
        elif target.year is not None and incoming_year is not None and target.year != incoming_year:
            PaperMerger._add_conflict(target, "year: conflicting values retained first")

    @staticmethod
    def _max_optional(left, right):
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)

    @staticmethod
    def _merge_authors(left: list[Author], right: list[Author]) -> list[Author]:
        result = [deepcopy(author) for author in left]
        positions = {normalize_title(author.name): index for index, author in enumerate(result)}
        for author in right:
            key = normalize_title(author.name)
            if key not in positions:
                positions[key] = len(result)
                result.append(deepcopy(author))
                continue
            current = result[positions[key]]
            current.affiliations = PaperMerger._unique(current.affiliations + author.affiliations)
            current.orcid = current.orcid or author.orcid
            if current.corresponding is None:
                current.corresponding = author.corresponding
        return result

    @staticmethod
    def _merge_sources(left: list[SourceRecord], right: list[SourceRecord]) -> list[SourceRecord]:
        result = [deepcopy(record) for record in left]
        for record in right:
            # Equal snapshots are redundant; the same provider/id may also
            # supply a different raw response, which must remain auditable.
            if record not in result:
                result.append(deepcopy(record))
        return result

    @staticmethod
    def _unique(values):
        result = []
        seen = set()
        for value in values:
            key = str(value)
            if key not in seen:
                result.append(value)
                seen.add(key)
        return result

    @staticmethod
    def _add_conflict(paper: Paper, message: str) -> None:
        if message not in paper.metadata_conflicts:
            paper.metadata_conflicts.append(message)
