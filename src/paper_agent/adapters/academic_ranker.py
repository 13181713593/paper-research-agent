"""Two-stage academic ranking with an optional locally cached CrossEncoder."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from paper_agent.domain.models import (
    CodeAvailability,
    Paper,
    PublicationStatus,
    ResearchIdea,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)
PROMPT_VERSION = "academic-intent-v1"
_STOP = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "with",
    "using",
    "study",
    "research",
    "paper",
    "papers",
    "find",
    "recent",
    "latest",
}
_FOCUS = {
    "recency": r"\b(recent|latest|newest)\b|最新|近年",
    "citations": r"\b(highly cited|citations?)\b|高引用|引用量",
    "venue": r"\b(top venues?|prestigious)\b|顶会|顶刊",
    "code": r"\b(open.source|reproducib\w*|code)\b|开源|代码|可复现",
}


def _tokens(text: str) -> set[str]:
    english = set(re.findall(r"[a-z0-9]+", text.casefold())) - _STOP
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return english | {run[i : i + 2] for run in chinese for i in range(max(1, len(run) - 1))}


def _coverage(query: str, passage: str) -> float:
    tokens = _tokens(query)
    return len(tokens & _tokens(passage)) / len(tokens) if tokens else 0.0


def _contains(term: str, text: str) -> bool:
    if re.fullmatch(r"[a-zA-Z0-9 ]+", term):
        return bool(re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))
    return term.casefold() in text.casefold()


class AcademicIntentRewriter:
    """Conservative rule-based extraction; explicit structured constraints win."""

    async def rewrite(self, idea: ResearchIdea) -> ResearchIdea:
        rewritten = idea.model_copy(deep=True)
        for focus, pattern in _FOCUS.items():
            if (
                re.search(pattern, idea.text, re.IGNORECASE)
                and focus not in rewritten.ranking_focus
            ):
                rewritten.ranking_focus.append(focus)
        # Only unambiguous, quoted exclusion phrases become hard constraints.
        for term in re.findall(
            r'(?:exclude|excluding|排除|不要)\s*["“]([^"”]+)["”]', idea.text, re.IGNORECASE
        ):
            if term not in rewritten.excluded_terms:
                rewritten.excluded_terms.append(term)
        return rewritten


def build_ranking_prompt(idea: ResearchIdea) -> str:
    return (
        f"Research question: {idea.text}\n"
        f"Required concepts: {', '.join(idea.required_terms) or 'none specified'}\n"
        f"Excluded concepts: {', '.join(idea.excluded_terms) or 'none specified'}\n"
        f"Ranking emphasis: {', '.join(idea.ranking_focus) or 'topic relevance'}\n"
        "Prioritize papers directly addressing the question and required concepts."
    )


@dataclass(frozen=True, slots=True)
class AcademicRankingConfig:
    model_name: str | None = None
    local_files_only: bool = True
    batch_size: int = 16
    reference_year: int | None = None


class BgeReranker:
    """Load once; serialize inference and expose failures in each score explanation."""

    def __init__(self, config: AcademicRankingConfig):
        self.config = config
        self._model = None
        self._failed: str | None = None
        self._lock = threading.Lock()

    def predict(self, query: str, passages: list[str]) -> tuple[list[float] | None, str]:
        if not self.config.model_name:
            return None, "lexical-fallback (no reranker configured)"
        with self._lock:
            if self._failed:
                return None, self._failed
            try:
                import torch
                from sentence_transformers import CrossEncoder

                if self._model is None:
                    self._model = CrossEncoder(
                        self.config.model_name,
                        local_files_only=self.config.local_files_only,
                        max_length=512,
                    )
                logits = self._model.predict(
                    [(query, passage) for passage in passages],
                    batch_size=self.config.batch_size,
                    activation_fn=torch.nn.Identity(),
                    show_progress_bar=False,
                )
                values = [float(value) for value in logits]
                if len(values) != len(passages) or not all(math.isfinite(v) for v in values):
                    raise ValueError("Reranker must return one finite logit per paper")
                scores = [1 / (1 + math.exp(-max(-60, min(60, v)))) for v in values]
                return scores, f"CrossEncoder:{self.config.model_name}; sigmoid(logit)"
            # Optional third-party model failures must preserve offline ranking.
            except Exception as exc:  # noqa: BLE001
                self._failed = f"lexical-fallback (reranker failed: {type(exc).__name__})"
                logger.warning("Academic reranker unavailable: %s", exc)
                return None, self._failed


def _metadata(paper: Paper) -> str:
    return "\n".join([paper.title, paper.abstract or "", " ".join(paper.keywords)])


def _evidence(paper: Paper, idea: ResearchIdea) -> tuple[str, float, str]:
    if paper.fulltext.markdown_path:
        try:
            with Path(paper.fulltext.markdown_path).open(encoding="utf-8") as stream:
                text = stream.read(120_000)
            if text.strip():
                paragraphs = re.split(r"\n\s*\n", text)
                paragraphs.sort(key=lambda p: _coverage(idea.text, p), reverse=True)
                excerpt = "\n\n".join(paragraphs[:6])[:6000]
                quality = paper.fulltext.parse_quality
                return excerpt, quality if quality is not None else 0.5, "markdown"
        except (OSError, UnicodeError):
            pass
    return (
        paper.abstract or "",
        0.35 if paper.abstract else 0.0,
        "abstract" if paper.abstract else "none",
    )


def _eligible(idea: ResearchIdea, paper: Paper, text: str) -> bool:
    if paper.is_retracted or paper.publication_status == PublicationStatus.RETRACTED:
        return False
    if any(_contains(term, text) for term in idea.excluded_terms):
        return False
    if not all(_contains(term, text) for term in idea.required_terms):
        return False
    published = paper.publication_date
    # A year-only record is compatible if its year overlaps the requested range.
    if idea.date_from:
        if published and published < idea.date_from:
            return False
        if not published and (paper.year is None or paper.year < idea.date_from.year):
            return False
    if idea.date_to:
        if published and published > idea.date_to:
            return False
        if not published and (paper.year is None or paper.year > idea.date_to.year):
            return False
    return True


def _signals(idea: ResearchIdea, paper: Paper, year: int) -> tuple[float, float, float]:
    recency = 0.0 if paper.year is None else 2 ** (-max(0, year - paper.year) / 5)
    venue = (
        " ".join(filter(None, [paper.venue.name, paper.venue.short_name])) if paper.venue else ""
    )
    # This is explicit venue preference, not an inferred peer-review/quality label.
    venue_match = float(bool(venue) and any(_contains(v, venue) for v in idea.preferred_venues))
    citations = min(1.0, math.log1p(paper.citation_count or 0) / math.log1p(1000))
    return recency, venue_match, citations


class AcademicEvidenceVerifier:
    async def verify(self, idea: ResearchIdea, paper: Paper) -> Paper:
        intent = await AcademicIntentRewriter().rewrite(idea)
        _, coverage, source = await asyncio.to_thread(_evidence, paper, intent)
        result = paper.model_copy(deep=True)
        result.scores.evidence_coverage = coverage
        result.scores.explanation = (
            f"Evidence source={source}; availability/parse quality={coverage:.3f}"
        )
        return result


class AcademicPreRanker:
    """Cheap metadata-only high-recall pool selection, with exclusion screening."""

    def __init__(self, config: AcademicRankingConfig | None = None):
        self.config = config or AcademicRankingConfig()
        self.rewriter = AcademicIntentRewriter()

    async def rank(self, idea: ResearchIdea, papers: Sequence[Paper], limit: int) -> list[Paper]:
        if limit < 1:
            raise ValueError("limit must be positive")
        intent = await self.rewriter.rewrite(idea)
        results = []
        for paper in papers:
            text = _metadata(paper)
            # Required concepts can appear only in full text; defer that check.
            screening = intent.model_copy(update={"required_terms": []})
            if not _eligible(screening, paper, text):
                continue
            relevance = 0.7 * _coverage(intent.text, text) + 0.3 * _coverage(
                intent.text, paper.title
            )
            required = sum(_contains(t, text) for t in intent.required_terms)
            match = required / len(intent.required_terms) if intent.required_terms else relevance
            recency, venue, citations = _signals(
                intent, paper, self.config.reference_year or date.today().year
            )
            result = paper.model_copy(deep=True)
            result.scores = ScoreBreakdown(
                lexical=relevance,
                intent_match=match,
                recency=recency,
                venue_quality=venue,
                citation_signal=citations,
                final=0.8 * relevance
                + 0.1 * match
                + 0.05 * recency
                + 0.03 * venue
                + 0.02 * citations,
                formula_version="academic-v1-pre",
                explanation=f"Metadata lexical ranking; prompt={PROMPT_VERSION}; focus={intent.ranking_focus}",
            )
            results.append(result)
        return sorted(results, key=lambda p: p.scores.final or 0, reverse=True)[:limit]


class AcademicFinalRanker(AcademicPreRanker):
    def __init__(self, config: AcademicRankingConfig | None = None):
        super().__init__(config)
        self.backend = BgeReranker(self.config)

    async def rank(self, idea: ResearchIdea, papers: Sequence[Paper], limit: int) -> list[Paper]:
        if limit < 1:
            raise ValueError("limit must be positive")
        intent = await self.rewriter.rewrite(idea)
        candidates = []
        for paper in papers:
            excerpt, quality, source = await asyncio.to_thread(_evidence, paper, intent)
            passage = f"{_metadata(paper)}\nEvidence:\n{excerpt}"
            if _eligible(intent, paper, passage):
                candidates.append((paper, passage, quality, source))
        if not candidates:
            return []
        scores, backend = await asyncio.to_thread(
            self.backend.predict, build_ranking_prompt(intent), [c[1] for c in candidates]
        )
        results = []
        for index, (paper, passage, quality, source) in enumerate(candidates):
            lexical = _coverage(intent.text, passage)
            relevance = scores[index] if scores is not None else lexical
            match = (
                sum(_contains(t, passage) for t in intent.required_terms)
                / len(intent.required_terms)
                if intent.required_terms
                else lexical
            )
            if "code" in intent.ranking_focus:
                code_match = float(paper.code.status == CodeAvailability.OPEN_SOURCE)
                match = 0.8 * match + 0.2 * code_match
            recency, venue, citations = _signals(
                intent, paper, self.config.reference_year or date.today().year
            )
            weights = [0.45, 0.25, 0.15, 0.07, 0.05, 0.03]
            for focus, slot in [("recency", 3), ("venue", 4), ("citations", 5)]:
                if focus in intent.ranking_focus:
                    weights[slot] += 0.02
                    weights[2] -= 0.02
            evidence = quality * lexical
            total = sum(
                w * s
                for w, s in zip(
                    weights, [relevance, match, evidence, recency, venue, citations], strict=True
                )
            )
            result = paper.model_copy(deep=True)
            result.scores = ScoreBreakdown(
                lexical=lexical,
                reranker=relevance if scores is not None else None,
                intent_match=match,
                evidence_coverage=evidence,
                recency=recency,
                venue_quality=venue,
                citation_signal=citations,
                final=min(1.0, total),
                formula_version="academic-v1-final",
                explanation=(
                    f"backend={backend}; prompt={PROMPT_VERSION}; evidence={source}; "
                    f"focus={intent.ranking_focus}; weights={weights}; "
                    f"relevance={relevance:.3f}, intent={match:.3f}, evidence={evidence:.3f}, "
                    f"recency={recency:.3f}, venue_preference={venue:.3f}, citations={citations:.3f}"
                ),
            )
            results.append(result)
        return sorted(results, key=lambda p: p.scores.final or 0, reverse=True)[:limit]
