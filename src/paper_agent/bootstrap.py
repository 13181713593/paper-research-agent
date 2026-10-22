"""Dependency composition root.

Group implementations are wired here and nowhere else. During parallel work,
each group can replace one adapter without changing the pipeline, MCP server,
or Streamlit page.

Assembly modes (``PAPER_AGENT_ENV``):
- ``demo`` (default): offline demo components, unchanged team baseline.
- ``retrieval``: retrieval group (A + B) production components for the
  retrieval segment — real query planner, four live sources (arXiv,
  Crossref, OpenAlex, Semantic Scholar) and the real merger. All later
  stages (ranking, full text, summarization, report) stay demo until
  groups 2 and 3 deliver their adapters.
- ``ranking``: demo components with the ranking group's academic
  pre-ranker, evidence verifier and final ranker swapped in.
"""

from __future__ import annotations

from dataclasses import replace

from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters.academic_ranker import (
    AcademicEvidenceVerifier,
    AcademicFinalRanker,
    AcademicPreRanker,
    AcademicRankingConfig,
)
from paper_agent.adapters.arxiv_retriever import ArxivRetriever, ArxivSearchConfig
from paper_agent.adapters.crossref_retriever import CrossrefRetriever
from paper_agent.adapters.europepmc_retriever import EuropePmcRetriever
from paper_agent.adapters.llm_concept_extractor import LLMConceptExtractor
from paper_agent.adapters.openalex_retriever import OpenAlexRetriever
from paper_agent.adapters.paper_merger import PaperMerger
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import PipelineDependencies, ResearchPipeline
from paper_agent.config import Settings, load_local_env


def build_retrieval_dependencies() -> PipelineDependencies:
    """Retrieval segment (group 1, sides A + B) production assembly.

    Real planner/retrievers/merger are swapped into the shared dependency
    graph; components owned by other groups are taken from the demo assembly
    and must be replaced by their owners later.
    """
    demo = build_demo_dependencies()
    # LLM-first concept extraction (translates Chinese ideas to English
    # search keywords). Inactive without PAPER_AGENT_LLM_API_KEY; degrades
    # transparently to the offline heuristic extractor otherwise.
    extractor = LLMConceptExtractor()
    return replace(
        demo,
        query_planner=HeuristicQueryPlanner(concept_extractor=extractor),
        retrievers=[
            ArxivRetriever(config=ArxivSearchConfig(max_retries=2)),
            CrossrefRetriever(),
            OpenAlexRetriever(),
            SemanticScholarRetriever(),
            EuropePmcRetriever(),
        ],
        merger=PaperMerger(),
    )


def build_agent(settings: Settings | None = None) -> PaperResearchAgent:
    load_local_env()  # Local secrets (.env is gitignored); real env wins.
    settings = settings or Settings.from_env()
    if settings.app_env == "demo":
        deps = build_demo_dependencies()
    elif settings.app_env == "retrieval":
        deps = build_retrieval_dependencies()
    elif settings.app_env == "ranking":
        deps = build_demo_dependencies()
        config = AcademicRankingConfig(
            model_name=settings.reranker_model,
            local_files_only=settings.reranker_local_files_only,
        )
        deps.pre_ranker = AcademicPreRanker(config)
        deps.verifier = AcademicEvidenceVerifier()
        deps.final_ranker = AcademicFinalRanker(config)
    else:
        raise RuntimeError(
            "Production adapters are not implemented yet. Complete group 1-3 adapters "
            "and replace this branch in paper_agent.bootstrap.build_agent()."
        )
    pipeline = ResearchPipeline(deps, max_concurrency=settings.max_concurrency)
    return PaperResearchAgent(pipeline)

