"""统一装配入口：保留各组联调模式，production 串起三组真实组件。"""

from __future__ import annotations

from dataclasses import replace

from paper_agent.adapters import build_demo_dependencies, build_processing_components
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
from paper_agent.adapters.openreview_retriever import OpenReviewRetriever
from paper_agent.adapters.paper_merger import PaperMerger
from paper_agent.adapters.query_planner import HeuristicQueryPlanner
from paper_agent.adapters.semantic_scholar_retriever import SemanticScholarRetriever
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import PipelineDependencies, ResearchPipeline
from paper_agent.config import Settings, load_local_env


def build_retrieval_dependencies(settings: Settings | None = None) -> PipelineDependencies:
    """第一组六源检索与去重；其他阶段保留演示组件供单组联调。"""
    settings = settings or Settings.from_env()
    extractor = LLMConceptExtractor(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=min(20.0, settings.request_timeout_seconds),
    )
    return replace(
        build_demo_dependencies(),
        query_planner=HeuristicQueryPlanner(concept_extractor=extractor, expand_queries=True),
        retrievers=[
            ArxivRetriever(config=ArxivSearchConfig(max_retries=2)),
            CrossrefRetriever(),
            OpenAlexRetriever(),
            SemanticScholarRetriever(),
            EuropePmcRetriever(),
            OpenReviewRetriever(),
        ],
        merger=PaperMerger(),
    )


def build_agent(settings: Settings | None = None) -> PaperResearchAgent:
    load_local_env()
    settings = settings or Settings.from_env()
    if settings.app_env not in {"demo", "retrieval", "ranking", "production"}:
        raise RuntimeError(
            f"Unknown PAPER_AGENT_ENV '{settings.app_env}'. Supported values: "
            "demo, retrieval, ranking, production (full research pipeline)."
        )
    deps = (
        build_retrieval_dependencies(settings)
        if settings.app_env in {"retrieval", "production"}
        else build_demo_dependencies()
    )
    if settings.app_env in {"ranking", "production"}:
        config = AcademicRankingConfig(
            model_name=settings.reranker_model,
            embedding_model=settings.embedding_model,
            local_files_only=settings.reranker_local_files_only,
        )
        deps.pre_ranker = AcademicPreRanker(config)
        deps.verifier = AcademicEvidenceVerifier()
        deps.final_ranker = AcademicFinalRanker(config)
    if settings.app_env == "production":
        components = build_processing_components(settings)
        deps.fulltext_processor = components.fulltext_processor
        deps.summarizer = components.summarizer
        deps.report_writer = components.report_writer
    pipeline = ResearchPipeline(
        deps,
        max_concurrency=settings.max_concurrency,
        source_timeout_seconds=settings.source_timeout_seconds,
    )
    return PaperResearchAgent(pipeline)


def build_processing_workbench(settings: Settings | None = None):
    """独立工作台仅处理指定论文，不执行检索或排序。"""
    from paper_agent.adapters.arxiv_metadata import ArxivMetadataEnricher
    from paper_agent.application.processing_workbench import ProcessingWorkbench

    load_local_env()
    settings = settings or Settings.from_env()
    metadata = ArxivMetadataEnricher(timeout_seconds=min(15.0, settings.request_timeout_seconds))
    return ProcessingWorkbench(
        lambda executor: build_processing_components(settings, parser_executor=executor),
        metadata_enricher=metadata.enrich,
    )
