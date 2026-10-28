"""Serializable stage snapshots, independent of mutable adapter objects and raw payloads."""

import os
import re

from paper_agent.domain.models import Paper, PaperSummary, ResearchReport, SearchPlan

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "token",
    "password",
    "email",
    "mailto",
    "signature",
    "policy",
    "key_pair_id",
    "expires",
}
_SENSITIVE_QUERY = re.compile(
    r"([?&](?:amp;)?(?:api_key|apikey|access_token|token|password|email|mailto|"
    r"signature|policy|key-pair-id|expires|x-amz-[\w-]+|x-goog-[\w-]+)=)"
    r"[^&\s\"'<>]+",
    re.IGNORECASE,
)


def redact_audit_value(value):
    """仅脱敏展示/导出副本；下载器仍使用原始 URL，不改动领域对象。"""
    if isinstance(value, dict):
        return {
            key: "[已隐藏]"
            if key.lower().replace("-", "_") in _SENSITIVE_KEYS
            else redact_audit_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_audit_value(item) for item in value]
    if isinstance(value, str):
        value = _SENSITIVE_QUERY.sub(r"\1[已隐藏]", value)
        for name, secret in os.environ.items():
            if (
                secret
                and len(secret) >= 6
                and any(
                    marker in name.upper() for marker in ("API_KEY", "TOKEN", "PASSWORD", "EMAIL")
                )
            ):
                value = value.replace(secret, "[已隐藏]")
    return value


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
