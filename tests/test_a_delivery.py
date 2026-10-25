from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from paper_agent.adapters.arxiv_retriever import ArxivRetriever, ArxivSearchConfig
from paper_agent.adapters.bibtex_mapper import BibTexMapper, BibTexMappingError
from paper_agent.adapters.paper_merger import PaperMerger
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import (
    AccessStatus,
    Author,
    CodeAsset,
    CodeAvailability,
    FullTextAsset,
    Paper,
    PaperSource,
    SearchPlan,
    SearchQuery,
    SourceRecord,
    Venue,
)


def plan_for(*queries: str, limit: int = 5) -> SearchPlan:
    return SearchPlan(
        intent_summary="retrieval test idea",
        queries=[SearchQuery(source=PaperSource.ARXIV, query=q, limit=limit) for q in queries],
    )


def result(**kwargs):
    base = dict(  # noqa: C408
        entry_id="http://arxiv.org/abs/2401.01234v2",
        title="A \n  Test: Paper!",
        authors=[SimpleNamespace(name="Doe, Jane"), SimpleNamespace(name="John Smith")],
        summary="A real abstract.",
        published=date(2024, 1, 2),
        updated=date(2024, 1, 4),
        doi="https://doi.org/10.1234/ABC",
        comment="12 pages",
        journal_ref="Journal of Testing",
        primary_category="cs.AI",
        categories=["cs.AI", "cs.IR"],
        pdf_url="https://arxiv.org/pdf/2401.01234v2",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_bibtex_mapper_maps_authors_abstract_and_provenance():
    bib = """@article{doe2024,
 title = {Evidence {Aware} Retrieval},
 author = {Doe, Jane and John Smith},
 year = {2024},
 doi = {https://doi.org/10.1234/ABC},
 journal = {Journal of Testing}
}"""
    paper = await BibTexMapper().to_paper(
        bib,
        source="arxiv",
        supplemental_metadata={
            "source_id": "2401.01234v2",
            "source_url": "https://arxiv.org/abs/2401.01234v2",
            "abstract": "A real abstract.",
            "arxiv_id": "2401.01234v2",
            "published": "2024-01-02T00:00:00Z",
            "pdf_url": "https://arxiv.org/pdf/2401.01234v2",
            "raw_metadata": {"provider": "fixture"},
        },
    )
    assert paper.title == "Evidence Aware Retrieval"
    assert [a.name for a in paper.authors] == ["Jane Doe", "John Smith"]
    assert paper.abstract == "A real abstract."
    assert paper.doi == "10.1234/abc"
    assert paper.arxiv_id == "2401.01234v2"
    assert paper.source_records[0].source == PaperSource.ARXIV
    assert paper.source_records[0].source_id == "2401.01234v2"
    assert paper.source_records[0].raw_metadata["provider"] == "fixture"


@pytest.mark.asyncio
async def test_bibtex_mapper_leaves_unknown_fields_none_and_rejects_multiple_entries():
    mapper = BibTexMapper()
    paper = await mapper.to_paper("@article{x, title={Only title}}", source="arxiv")
    assert paper.title == "Only title"
    assert paper.abstract is None
    assert paper.authors == []
    assert paper.publication_date is None
    with pytest.raises(BibTexMappingError):
        await mapper.to_paper("@article{x, title={A}} @article{y, title={B}}", source="arxiv")


@pytest.mark.asyncio
async def test_arxiv_retriever_uses_source_queries_limit_and_maps_results():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def results(self, search):
            self.calls.append(search)
            yield result(entry_id=f"https://arxiv.org/abs/2401.0123{len(self.calls)}v1")
            yield result(entry_id="https://arxiv.org/abs/2401.99999v1", title="Beyond limit")

    client = FakeClient()
    retriever = ArxivRetriever(
        mapper=BibTexMapper(), client=client, config=ArxivSearchConfig(delay_seconds=0)
    )
    papers = await retriever.search(plan_for("neural retrieval", "dense ranking", limit=1))
    assert len(papers) == 2
    assert len(client.calls) == 2
    assert all(p.source_records[0].source == PaperSource.ARXIV for p in papers)
    assert all(p.bibtex and p.arxiv_id for p in papers)


@pytest.mark.asyncio
async def test_arxiv_retriever_wraps_client_failures():
    class BrokenClient:
        def results(self, search):
            raise TimeoutError("network timeout")
            yield  # pragma: no cover

    retriever = ArxivRetriever(
        mapper=BibTexMapper(), client=BrokenClient(), config=ArxivSearchConfig(delay_seconds=0)
    )
    with pytest.raises(ProviderUnavailableError, match="network timeout"):
        await retriever.search(plan_for("failure"))


def paper(**kwargs) -> Paper:
    values = {"title": "Example Paper", "authors": [Author(name="Jane Doe")]}
    values.update(kwargs)
    return Paper(**values)


@pytest.mark.asyncio
async def test_merger_normalizes_doi_and_preserves_all_sources():
    left = paper(
        doi="https://doi.org/10.1000/XYZ",
        source_records=[SourceRecord(source=PaperSource.ARXIV, source_id="a")],
    )
    right = paper(
        doi="doi:10.1000/xyz",
        abstract="More complete",
        source_records=[SourceRecord(source=PaperSource.OPENALEX, source_id="b")],
    )
    merged = await PaperMerger().merge_and_deduplicate([left, right])
    assert len(merged) == 1
    assert merged[0].doi == "10.1000/xyz"
    assert merged[0].abstract == "More complete"
    assert {r.source_id for r in merged[0].source_records} == {"a", "b"}


def test_merger_strips_arxiv_versions_but_keeps_different_extensions():
    first = paper(title="A Method", arxiv_id="2401.01234v1")
    second = paper(title="A Method", arxiv_id="2401.01234v2")
    journal = paper(title="A Method: Extended Version", doi="10.1000/journal")
    result = asyncio.run(PaperMerger().merge_and_deduplicate([first, second, journal]))
    assert len(result) == 2
    assert any(p.title == "A Method" for p in result)


def test_merger_matches_arxiv_id_when_only_one_record_has_doi():
    preprint = paper(title="A Method", arxiv_id="2401.01234v1")
    enriched = paper(title="A Method", arxiv_id="2401.01234v2", doi="10.1000/method")
    result = asyncio.run(PaperMerger().merge_and_deduplicate([preprint, enriched]))
    assert len(result) == 1
    assert result[0].doi == "10.1000/method"


def test_merger_title_punctuation_is_duplicate_but_same_title_different_doi_is_not():
    first = paper(title="Neural-Retrieval: A Study", doi=None)
    second = paper(title="Neural Retrieval A Study", doi=None)
    different = paper(title="Neural Retrieval A Study", doi="10.1000/different")
    result = asyncio.run(PaperMerger().merge_and_deduplicate([first, second, different]))
    assert len(result) == 2


def test_merger_does_not_merge_similar_conference_and_journal_versions():
    conference = paper(
        title="Retrieval for Agents",
        venue=Venue(name="AI Conference", venue_type="conference"),
    )
    journal = paper(
        title="Retrieval for Agents", venue=Venue(name="AI Journal", venue_type="journal")
    )
    result = asyncio.run(PaperMerger().merge_and_deduplicate([conference, journal]))
    assert len(result) == 2


def test_merger_is_idempotent_and_does_not_mutate_inputs():
    original = paper(title="Input Paper")
    result = asyncio.run(PaperMerger().merge_and_deduplicate([original]))
    assert original.normalized_title is None
    again = asyncio.run(PaperMerger().merge_and_deduplicate(result))
    assert len(again) == 1
    assert len(again[0].source_records) == len(result[0].source_records)


def test_merger_keeps_distinct_external_ids_from_title_only_match():
    left = paper(openalex_id="W123")
    right = paper(openalex_id="W456")
    result = asyncio.run(PaperMerger().merge_and_deduplicate([left, right]))
    assert len(result) == 2


def test_merger_combines_fulltext_fields_without_losing_parser_output():
    target = paper(
        fulltext=FullTextAsset(
            markdown_path="artifacts/paper.md",
            parser_name="marker",
            parse_quality=0.9,
        )
    )
    incoming = paper(
        fulltext=FullTextAsset(
            pdf_url="https://example.org/paper.pdf",
            access_status=AccessStatus.OPEN_ACCESS,
        )
    )
    result = asyncio.run(PaperMerger().merge_and_deduplicate([target, incoming]))
    assert len(result) == 1
    merged = result[0].fulltext
    assert str(merged.pdf_url) == "https://example.org/paper.pdf"
    assert merged.markdown_path == "artifacts/paper.md"
    assert merged.parser_name == "marker"
    assert merged.parse_quality == 0.9
    assert merged.access_status is AccessStatus.OPEN_ACCESS


def test_merger_preserves_first_code_fields_and_records_conflicts():
    target = paper(
        code=CodeAsset(
            status=CodeAvailability.REPOSITORY_FOUND,
            repository_url="https://github.com/example/first",
        )
    )
    incoming = paper(
        code=CodeAsset(
            status=CodeAvailability.OPEN_SOURCE,
            repository_url="https://github.com/example/second",
            license="MIT",
        )
    )
    result = asyncio.run(PaperMerger().merge_and_deduplicate([target, incoming]))
    assert len(result) == 1
    merged = result[0]
    assert str(merged.code.repository_url) == "https://github.com/example/first"
    assert merged.code.license == "MIT"
    assert any("code.status" in item for item in merged.metadata_conflicts)
    assert any("code.repository_url" in item for item in merged.metadata_conflicts)
