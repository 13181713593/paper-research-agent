"""Paper research agent package."""

from .application.agent import PaperResearchAgent
from .application.pipeline import ResearchPipeline

__version__ = "0.2.0"

__all__ = ["PaperResearchAgent", "ResearchPipeline", "__version__"]
