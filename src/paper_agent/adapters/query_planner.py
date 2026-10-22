"""Query planner: rewrite a research idea into per-source keyword queries.

Contract notes (ports/contracts.py + docs/API_CONTRACTS.md):
- ``plan`` must not perform network I/O in the DEFAULT assembly; the LLM
  extractor is opt-in and degrades to the heuristic extractor when the LLM
  is unreachable, so ``plan`` never fails because of the extractor.
- Queries are space-separated keywords; source-specific API syntax is each
  retriever's job.
"""

from __future__ import annotations

from typing import Protocol

from paper_agent.adapters._concept_extraction import (
    ExtractedConcepts,
    HeuristicConceptExtractor,
)
from paper_agent.domain.models import (
    PaperSource,
    ResearchIdea,
    SearchPlan,
    SearchQuery,
)

# Sources our group's retrievers implement; the planner emits one query per
# source so each retriever picks exactly its own entry from the plan.
_PLANNED_SOURCES = (
    PaperSource.ARXIV,
    PaperSource.OPENALEX,
    PaperSource.CROSSREF,
    PaperSource.SEMANTIC_SCHOLAR,
    PaperSource.EUROPE_PMC,
)


class ConceptExtractor(Protocol):
    def extract(self, text: str) -> ExtractedConcepts: ...


class HeuristicQueryPlanner:
    """Deterministic planner over an injectable concept extractor.

    Default extractor is offline rule-based. Production assembly may inject
    ``LLMConceptExtractor`` (which itself falls back to rules on failure).
    """

    def __init__(self, concept_extractor: ConceptExtractor | None = None):
        self.concept_extractor = concept_extractor or HeuristicConceptExtractor()

    async def plan(self, idea: ResearchIdea, limit_per_source: int) -> SearchPlan:
        extracted = self.concept_extractor.extract(idea.text)
        concepts = list(extracted.concepts)
        if not concepts:
            # Extraction found nothing usable (e.g. Chinese input with no
            # lexicon hits and no LLM configured). Pass the raw text through
            # so OpenAlex/Crossref can still match it; never silently search
            # an unrelated default topic like "machine learning".
            concepts = [" ".join(idea.text.split())]

        for term in idea.required_terms:
            if term not in concepts:
                concepts.append(term)

        synonyms = list(extracted.synonyms)
        # Relevance-search sources (Crossref/OpenAlex/Semantic Scholar)
        # handle multi-keyword queries well; take the plain joined form.
        base_query = " ".join(concepts)
        # arXiv's Boolean API ANDs every term, so each extra keyword slashes
        # recall (verified: 3 phrases -> 0 hits, top phrase alone -> 76).
        # LLM concepts are importance-ordered, so arXiv searches only the
        # top concept as a quoted phrase — the same topic at the right
        # granularity for that API. (See test hits in git history.)
        top = concepts[0]
        arxiv_query = f'"{top}"' if " " in top else top
        # Filter keys follow the retrieval group's shared protocol
        # (see ArxivRetriever): only date_from/date_to/excluded_terms/
        # required_terms are recognized. Synonyms live on the plan itself.
        filters: dict[str, list[str]] = {}
        if idea.required_terms:
            # arXiv queries carry only the top concept (recall), so required
            # terms reach arXiv through the filter the adapter ANDs in.
            filters["required_terms"] = list(idea.required_terms)
        if idea.excluded_terms:
            filters["excluded_terms"] = list(idea.excluded_terms)
        queries = []
        for source in _PLANNED_SOURCES:
            queries.append(
                SearchQuery(
                    source=source,
                    query=arxiv_query if source == PaperSource.ARXIV else base_query,
                    limit=limit_per_source,
                    filters=filters,
                )
            )

        inclusion = [
            "Paper metadata matches at least one core concept of the idea",
            "Title or abstract available in English",
        ]
        if idea.required_terms:
            inclusion.append(f"Contains required terms: {', '.join(idea.required_terms)}")

        exclusion = list(idea.excluded_terms) or [
            "Datasets and benchmarks without methodological contribution",
        ]

        return SearchPlan(
            intent_summary=idea.text,
            queries=queries,
            synonyms=synonyms,
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
        )
