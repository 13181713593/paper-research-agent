"""MCP v2 server exposing the main-agent workflow as typed tools.

Keep tools coarse-grained for normal hosts: large PDF/Markdown payloads remain
inside the service and only structured results cross the MCP boundary.
"""

from __future__ import annotations

from mcp.server import MCPServer

from paper_agent.bootstrap import build_agent
from paper_agent.domain.models import (
    PipelineContract,
    PipelineRequest,
    PipelineResult,
    ResearchIdea,
)

mcp = MCPServer("paper-research-agent")
agent = build_agent()


@mcp.tool()
def get_pipeline_contract() -> PipelineContract:
    """Return workflow stages and stable integration constraints."""
    return agent.contract()


@mcp.tool()
async def research_papers(
    research_idea: str,
    top_k: int = 5,
    pre_rank_pool_size: int = 20,
    require_fulltext: bool = False,
) -> PipelineResult:
    """Retrieve, verify, rank, summarize, and synthesize papers for a research idea.

    Args:
        research_idea: A specific research question or proposed direction.
        top_k: Number of finally selected papers (1-20).
        pre_rank_pool_size: Metadata-ranked candidates sent to full-text processing.
        require_fulltext: Exclude records that cannot produce parsed full text.

    Returns:
        A structured result containing papers, score explanations, summaries,
        report, warnings, and stage timing. The bundled implementation is an
        offline demo; replace adapters in ``build_demo_dependencies``.
    """
    request = PipelineRequest(
        idea=ResearchIdea(text=research_idea),
        top_k=top_k,
        pre_rank_pool_size=pre_rank_pool_size,
        require_fulltext=require_fulltext,
    )
    return await agent.research(request)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
