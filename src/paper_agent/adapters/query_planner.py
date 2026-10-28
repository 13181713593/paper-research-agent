"""Query planner: rewrite a research idea into per-source keyword queries.

Contract notes (ports/contracts.py + docs/API_CONTRACTS.md):
- ``plan`` must not perform network I/O in the DEFAULT assembly; the LLM
  extractor is opt-in and degrades to the heuristic extractor when the LLM
  is unreachable, so ``plan`` never fails because of the extractor.
- Queries are space-separated keywords; source-specific API syntax is each
  retriever's job.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Protocol

from paper_agent.adapters._concept_extraction import (
    ExtractedConcepts,
    HeuristicConceptExtractor,
    _synonyms_for,
)
from paper_agent.adapters.query_expansion import search_variants
from paper_agent.adapters.query_rewriting import chinese_query, focused_variants, format_expression
from paper_agent.domain.models import (
    PaperSource,
    ResearchIdea,
    SearchPlan,
    SearchQuery,
)
from paper_agent.domain.publication_policy import years_before

# Sources our group's retrievers implement; the planner emits one query per
# source so each retriever picks exactly its own entry from the plan.
_PLANNED_SOURCES = (
    PaperSource.ARXIV,
    PaperSource.OPENALEX,
    PaperSource.CROSSREF,
    PaperSource.SEMANTIC_SCHOLAR,
    PaperSource.EUROPE_PMC,
    PaperSource.OPENREVIEW,
)


class ConceptExtractor(Protocol):
    def extract(self, text: str) -> ExtractedConcepts: ...


class HeuristicQueryPlanner:
    """Deterministic planner over an injectable concept extractor.

    Default extractor is offline rule-based. Production assembly may inject
    ``LLMConceptExtractor`` (which itself falls back to rules on failure).
    """

    def __init__(self, concept_extractor: ConceptExtractor | None = None, *, expand_queries=False):
        self.concept_extractor = concept_extractor or HeuristicConceptExtractor()
        self.expand_queries = expand_queries

    async def plan(self, idea: ResearchIdea, limit_per_source: int) -> SearchPlan:
        extracted = await asyncio.to_thread(self.concept_extractor.extract, idea.text)
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

        if self.expand_queries:
            return self._expanded_plan(idea, concepts, extracted.synonyms, limit_per_source)

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
        for name in ("date_from", "date_to"):
            if getattr(idea, name):
                filters[name] = getattr(idea, name).isoformat()
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
            concepts=concepts,
        )

    def _expanded_plan(self, idea, concepts, synonyms, limit):
        # Keep phrases intact; adding every suggested term with AND destroys recall.
        chinese = bool(re.search(r"[\u4e00-\u9fff]", idea.text))
        local = HeuristicConceptExtractor().extract(idea.text)
        english = [
            c
            for c in dict.fromkeys([*concepts, *local.concepts])
            if re.search("[a-zA-Z]", c) and not re.search(r"[\u4e00-\u9fff]", c)
        ]
        if not chinese and len(idea.text.split()) <= 10:
            english.insert(0, idea.text.strip())
        english = [
            term
            for term in dict.fromkeys(english)
            if not any(
                term.casefold() != other.casefold()
                and len(term.split()) == 1
                and term.casefold() in other.casefold().split()
                for other in english
            )
            and (term not in idea.required_terms or term in local.concepts)
        ]
        candidates = list(
            dict.fromkeys(
                [*synonyms, *local.synonyms, *_synonyms_for([c.casefold() for c in english])]
            )
        )
        variants, term_notes = search_variants(english, candidates)
        variants = focused_variants(idea.text, english, variants, maximum=3)
        chinese_term = chinese_query(idea.text) if chinese else None
        # Known English concepts carry long-input intent; do not additionally
        # send the entire Chinese instruction as an exact phrase.
        if variants and len(idea.text) > 40:
            chinese_term = None
        if not variants and not chinese_term:
            chinese_term = " ".join(idea.text.split())[:120]
        for original in term_notes:
            term_notes[original] = (
                "用于短检索式或其主题约束"
                if any(original.casefold() in v.casefold() for v in variants)
                else "未单独请求：每个来源最多3条英文表达，保留核心主题与应用约束"
            )
        filters = {}
        for name in ("required_terms", "excluded_terms"):
            if getattr(idea, name):
                filters[name] = list(getattr(idea, name))
        for name in ("date_from", "date_to"):
            if getattr(idea, name):
                filters[name] = getattr(idea, name).isoformat()
        queries = []
        window_to = idea.date_to or datetime.now(UTC).date()
        window_from = idea.date_from or years_before(window_to, 5)
        if window_from > window_to:
            raise ValueError("检索开始日期不能晚于结束日期")
        recent_from = max(window_from, idea.recent_from or years_before(window_to, 2))
        recent_from = min(recent_from, window_to)
        recent_budget = min(limit, math.ceil(limit * idea.recent_share))
        cohorts = [(recent_from, window_to, recent_budget, "近2年优先")]
        if recent_budget == 0:
            cohorts = [(window_from, window_to, limit, "所选时间范围")]
        elif window_from < recent_from and limit > recent_budget:
            cohorts.append(
                (window_from, recent_from - timedelta(days=1), limit - recent_budget, "近3–5年补充")
            )
        elif window_from == recent_from:
            cohorts = [(window_from, window_to, limit, "所选时间范围")]
        for source in _PLANNED_SOURCES:
            # Execute alternatives independently and union their results. This
            # works even for sources without reliable Boolean OR syntax.
            if source in (PaperSource.ARXIV, PaperSource.EUROPE_PMC):
                # arXiv's Boolean API ANDs every term, and Europe PMC's
                # TITLE_ABS expression joins phrases with AND, so each query for
                # these boolean sources must be a single phrase. Use the
                # importance-ordered core concepts (plus the Chinese variant)
                # directly instead of the combined variants, so qualifiers like
                # "edge computing" stay searchable without AND collapse or
                # dropped qualifiers.
                terms = [(c, "en") for c in english]
                if chinese_term:
                    terms.append((chinese_term, "zh" if chinese else "en"))
                if not terms:
                    terms = [(idea.text.strip(), "en")]
            else:
                terms = [(v, "en") for v in variants]
                if chinese_term:
                    terms.append((chinese_term, "zh" if chinese else "en"))
                if not terms:
                    terms = [(idea.text.strip(), "en")]
            expanded: list[tuple[str, str]] = []
            for term, language in terms:
                if idea.required_terms and source != PaperSource.ARXIV:
                    term += " " + " ".join(idea.required_terms)
                expanded.append((format_expression(term, source.value), language))
            expanded = list(dict.fromkeys(expanded))

            if source == PaperSource.ARXIV:
                strategy = "学术短语匹配；多检索式取并集"
            elif source == PaperSource.EUROPE_PMC:
                strategy = "限定标题与摘要，避免生物医学泛词召回"
            elif source == PaperSource.OPENALEX:
                strategy = "短语相关性检索"
            elif source == PaperSource.OPENREVIEW:
                strategy = "OpenReview 标题、摘要与关键词检索"
            else:
                strategy = "题名与学术元数据相关性检索"

            total_en = sum(1 for _, language in expanded if language == "en")
            for expression, language in expanded:
                for start, end, budget, window_label in cohorts:
                    if budget < 1:
                        continue
                    queries.append(
                        SearchQuery(
                            source=source,
                            query=expression,
                            language=language,
                            strategy=strategy
                            + (
                                "；时间窗口内按相关性检索"
                                if source == PaperSource.CROSSREF
                                else "；按发表时间从新到旧"
                            ),
                            # Do not lose English recall slots when the Chinese
                            # variant has no records in an English-dominant index.
                            limit=max(
                                1,
                                math.ceil(
                                    budget / (total_en if language == "en" else len(expanded))
                                ),
                            ),
                            filters={
                                **filters,
                                "date_from": start.isoformat(),
                                "date_to": end.isoformat(),
                            },
                            sort_by="relevance"
                            if source == PaperSource.CROSSREF
                            else "publication_date",
                            time_window=window_label,
                        )
                    )
        return SearchPlan(
            intent_summary=idea.text,
            queries=queries,
            concepts=english,
            synonyms=candidates,
            query_terms=variants + ([chinese_term] if chinese_term else []),
            term_notes=term_notes,
            source_limit=limit,
            date_from=window_from,
            date_to=window_to,
            recent_from=recent_from,
            recent_share=idea.recent_share,
            planner_mode=getattr(self.concept_extractor, "last_mode", "heuristic"),
            inclusion_criteria=["标题、摘要或全文与核心研究主题相关；不以泛词重叠代替主题匹配"],
            exclusion_criteria=list(idea.excluded_terms),
            warnings=(
                [getattr(self.concept_extractor, "last_error", None)]
                if getattr(self.concept_extractor, "last_error", None)
                else []
            )
            + (
                ["当前未能生成英文检索词；请检查查询规划模型，或补充英文关键词。"]
                if chinese and not variants
                else []
            ),
        )
