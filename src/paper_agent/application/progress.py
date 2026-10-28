"""Serializable stage snapshots, independent of mutable adapter objects and raw payloads."""

from paper_agent.domain.models import Paper, PaperSummary, ResearchReport, SearchPlan


def paper_snapshot(paper: Paper) -> dict:
    url = paper.landing_page_url or paper.fulltext.landing_page_url
    if not url:
        url = next(
            (record.source_url for record in paper.source_records if record.source_url), None
        )
    return {
        "paper_id": str(paper.paper_id),
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "institutions": list(
            dict.fromkeys(
                [*paper.institutions, *(i for a in paper.authors for i in a.affiliations)]
            )
        ),
        "author_affiliations": [
            {"name": a.name, "affiliations": a.affiliations} for a in paper.authors
        ],
        "citation_count": paper.citation_count,
        "metadata_retrieved_at": max(
            (r.retrieved_at.isoformat() for r in paper.source_records), default=None
        ),
        "year": paper.year,
        "venue": paper.venue.name if paper.venue else None,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url": str(url) if url else (f"https://doi.org/{paper.doi}" if paper.doi else None),
        "sources": list(dict.fromkeys(record.source.value for record in paper.source_records)),
        "abstract": paper.abstract,
        "scores": paper.scores.model_dump(mode="json"),
        "fulltext": paper.fulltext.model_dump(mode="json"),
        "metadata_conflicts": list(paper.metadata_conflicts),
    }


def output_snapshot(value: object) -> dict:
    if isinstance(value, SearchPlan):
        return {"plan": value.model_dump(mode="json")}
    if isinstance(value, ResearchReport):
        return {"report": value.model_dump(mode="json")}
    if isinstance(value, Paper):
        return {"papers": [paper_snapshot(value)]}
    if isinstance(value, PaperSummary):
        return {"summaries": [value.model_dump(mode="json")]}
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], PaperSummary):
            return {"summaries": [summary.model_dump(mode="json") for summary in value]}
        return {"papers": [paper_snapshot(paper) for paper in value]}
    return {}


def output_count(value: object) -> int | None:
    if isinstance(value, SearchPlan):
        return len(value.queries)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (Paper, PaperSummary, ResearchReport)):
        return 1
    return None
