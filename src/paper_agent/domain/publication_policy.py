"""Rolling publication windows and a minimum recent-paper quota."""

import math
from datetime import date


def years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:  # February 29 in a non-leap target year.
        return value.replace(year=value.year - years, day=28)


def is_recent(paper, cutoff: date) -> bool:
    if paper.publication_date:
        return paper.publication_date >= cutoff
    # A boundary-year-only record cannot establish that it is after the exact cutoff.
    return paper.year is not None and paper.year > cutoff.year


def reserve_recent(papers, limit, cutoff=None, share=0.7):
    """Preserve input relevance order, reserving slots only among eligible candidates."""
    if cutoff is None:
        return list(papers[:limit])
    recent = [p for p in papers if is_recent(p, cutoff)]
    reserved = {p.paper_id for p in recent[: math.ceil(limit * share)]}
    selected = set(reserved)
    for paper in papers:
        if len(selected) >= limit:
            break
        selected.add(paper.paper_id)
    return [p for p in papers if p.paper_id in selected]
