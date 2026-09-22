"""Dependency composition root.

Group implementations are wired here and nowhere else. During parallel work,
each group can replace one adapter without changing the pipeline, MCP server,
or Streamlit page.
"""

from paper_agent.adapters import build_demo_dependencies
from paper_agent.application.agent import PaperResearchAgent
from paper_agent.application.pipeline import ResearchPipeline
from paper_agent.config import Settings


def build_agent(settings: Settings | None = None) -> PaperResearchAgent:
    settings = settings or Settings.from_env()
    if settings.app_env != "demo":
        raise RuntimeError(
            "Production adapters are not implemented yet. Complete group 1-3 adapters "
            "and replace this branch in paper_agent.bootstrap.build_agent()."
        )
    pipeline = ResearchPipeline(
        build_demo_dependencies(), max_concurrency=settings.max_concurrency
    )
    return PaperResearchAgent(pipeline)

