"""Dependency composition root.

Group implementations are wired here and nowhere else. During parallel work,
each group can replace one adapter without changing the pipeline, MCP server,
or Streamlit page.
"""

from paper_agent.adapters import build_demo_dependencies
from paper_agent.adapters.academic_ranker import (
    AcademicEvidenceVerifier,
    AcademicFinalRanker,
    AcademicPreRanker,
    AcademicRankingConfig,
)
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.config import Settings


def build_agent(settings: Settings | None = None) -> PaperResearchAgent:
    settings = settings or Settings.from_env()
    if settings.app_env not in {"demo", "ranking"}:
        raise RuntimeError(
            "Production adapters are not implemented yet. Complete group 1-3 adapters "
            "and replace this branch in paper_agent.bootstrap.build_agent()."
        )
    deps = build_demo_dependencies()
    if settings.app_env == "ranking":
        config = AcademicRankingConfig(
            model_name=settings.reranker_model,
            local_files_only=settings.reranker_local_files_only,
        )
        deps.pre_ranker = AcademicPreRanker(config)
        deps.verifier = AcademicEvidenceVerifier()
        deps.final_ranker = AcademicFinalRanker(config)
    pipeline = ResearchPipeline(deps, max_concurrency=settings.max_concurrency)
    return PaperResearchAgent(pipeline)

