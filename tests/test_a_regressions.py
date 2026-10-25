"""Regression tests for the A handoff; inputs below are synthetic fixtures."""

import asyncio
from itertools import permutations

import pytest

from paper_agent.adapters import BibTexMapper, BibTexMappingError, PaperMerger
from paper_agent.adapters.retrieval_common import normalize_doi
from paper_agent.domain.models import Paper, PublicationStatus, SourceRecord, Venue


def run(coroutine):
    return asyncio.run(coroutine)


def test_assignment_doi_org_without_scheme_matches_bare_doi():
    normalized = normalize_doi("doi.org/10.1000/EXAMPLE")
    left = Paper(title="Audit paper", doi="doi.org/10.1000/EXAMPLE")
    right = Paper(title="Audit paper", doi="10.1000/example")
    merged = run(PaperMerger().merge_and_deduplicate([left, right]))
    print({"normalized": normalized, "merged_count": len(merged)})
    assert normalized == "10.1000/example"
    assert len(merged) == 1


def test_shared_identifier_bridge_merges_all_three_orders():
    a = Paper(title="Audit paper", doi="10.1000/bridge")
    b = Paper(title="Audit paper", arxiv_id="2401.01234v1")
    c = Paper(title="Audit paper", doi="10.1000/bridge", arxiv_id="2401.01234v2")
    observations = []
    for order in permutations(range(3)):
        inputs = [[a, b, c][i] for i in order]
        first = run(PaperMerger().merge_and_deduplicate(inputs))
        second = run(PaperMerger().merge_and_deduplicate(first))
        observations.append(
            {"order": order, "first_count": len(first), "second_count": len(second)}
        )
    print(observations)
    assert all(row["first_count"] == 1 for row in observations)


async def conference_and_journal_with_partial_supplement():
    mapper = BibTexMapper()
    conference = await mapper.to_paper(
        "@inproceedings{conf,title={Audit Retrieval},author={Jane Doe},"
        "year={2024},booktitle={Audit AI Conference}}",
        source="crossref",
        supplemental_metadata={"source_id": "conf", "venue": {"name": "Audit AI Conference"}},
    )
    journal = await mapper.to_paper(
        "@article{journal,title={Audit Retrieval},author={Jane Doe},"
        "year={2024},journal={Audit AI Journal}}",
        source="openalex",
        supplemental_metadata={"source_id": "journal", "venue": {"name": "Audit AI Journal"}},
    )
    return conference, journal


def test_partial_venue_supplement_preserves_known_bibtex_type():
    conference, journal = run(conference_and_journal_with_partial_supplement())
    print({"conference": conference.venue.model_dump(), "journal": journal.venue.model_dump()})
    assert conference.venue.venue_type == "conference"
    assert journal.venue.venue_type == "journal"


def test_conference_and_journal_remain_separate_after_mapping():
    papers = run(conference_and_journal_with_partial_supplement())
    merged = run(PaperMerger().merge_and_deduplicate(papers))
    print(
        {
            "input_count": len(papers),
            "output_count": len(merged),
            "conflicts": [p.metadata_conflicts for p in merged],
        }
    )
    assert len(merged) == 2


def test_merge_fills_known_publication_status_and_project_url():
    left = Paper(title="Audit paper", doi="10.1000/metadata")
    right = Paper(
        title="Audit paper",
        doi="10.1000/metadata",
        publication_status=PublicationStatus.PUBLISHED,
        project_page_url="https://example.org/audit-project",
        source_records=[SourceRecord(source="crossref", source_id="10.1000/metadata")],
    )
    merged = run(PaperMerger().merge_and_deduplicate([left, right]))[0]
    print(
        {
            "publication_status": merged.publication_status.value,
            "project_page_url": str(merged.project_page_url),
            "conflicts": merged.metadata_conflicts,
        }
    )
    assert merged.publication_status == PublicationStatus.PUBLISHED
    assert str(merged.project_page_url) == "https://example.org/audit-project"


def test_unknown_institutions_null_is_treated_as_absent():
    paper = run(
        BibTexMapper().to_paper(
            "@misc{audit,title={Audit paper}}",
            source="crossref",
            supplemental_metadata={"institutions": None},
        )
    )
    assert paper.institutions == []


@pytest.mark.parametrize(
    "value",
    [
        "doi.org/10.1000/ABC",
        "DX.DOI.ORG/10.1000/ABC",
        "https://doi.org/10.1000%2FABC?tracking=1#section",
        "doi:10.1000/ABC",
    ],
)
def test_supported_doi_forms_keep_identifier_in_mapper(value):
    paper = run(
        BibTexMapper().to_paper("@misc{x,title={Fixture},doi={" + value + "}}", source="crossref")
    )
    assert paper.doi == "10.1000/abc"


@pytest.mark.parametrize(
    "value",
    [
        "doi.org.evil.example/10.1000/abc",
        "https://doi.org@evil.example/10.1000/abc",
        "doi.org/not-a-doi",
        "10.1000/abc extra",
    ],
)
def test_doi_normalization_does_not_accept_lookalike_domains(value):
    assert normalize_doi(value) is None


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_bridge_never_overrides_conflicting_dois(order):
    records = [
        Paper(title="Fixture", doi="10.1000/first", arxiv_id="2401.01234v1"),
        Paper(title="Fixture", doi="10.1000/second", arxiv_id="2401.01234v2"),
        Paper(title="Fixture", arxiv_id="2401.01234v3"),
    ]
    merged = run(PaperMerger().merge_and_deduplicate([records[i] for i in order]))
    assert len(merged) == 2
    assert {p.doi for p in merged} == {"10.1000/first", "10.1000/second"}


def test_bridge_preserves_all_provenance_and_input_values():
    records = [
        Paper(
            title="Fixture",
            doi="10.1000/bridge",
            source_records=[SourceRecord(source="crossref", source_id="doi")],
        ),
        Paper(
            title="Fixture",
            arxiv_id="2401.01234v1",
            source_records=[SourceRecord(source="arxiv", source_id="arxiv")],
        ),
        Paper(
            title="Fixture",
            doi="10.1000/bridge",
            arxiv_id="2401.01234v2",
            source_records=[SourceRecord(source="openalex", source_id="bridge")],
        ),
    ]
    before = [p.model_dump() for p in records]
    merged = run(PaperMerger().merge_and_deduplicate(records))
    assert len(merged) == 1
    assert {s.source_id for s in merged[0].source_records} == {"doi", "arxiv", "bridge"}
    assert [p.model_dump() for p in records] == before
    repeated = run(PaperMerger().merge_and_deduplicate(merged + records))
    assert len(repeated) == 1
    assert len(repeated[0].source_records) == 3


def test_partial_venue_supplement_keeps_other_bibtex_fields():
    paper = run(
        BibTexMapper().to_paper(
            "@article{x,title={Fixture},journal={Known Journal},volume={7},pages={10--20}}",
            source="crossref",
            supplemental_metadata={"venue": {"name": "Journal Name"}},
        )
    )
    assert paper.venue.name == "Journal Name"
    assert paper.venue.venue_type == "journal"
    assert paper.venue.volume == "7"
    assert paper.venue.pages == "10--20"


def test_conflicting_venue_type_does_not_erase_bibtex_evidence():
    paper = run(
        BibTexMapper().to_paper(
            "@inproceedings{x,title={Fixture},booktitle={Known Conference}}",
            source="crossref",
            supplemental_metadata={"venue": {"venue_type": "journal"}},
        )
    )
    assert paper.venue.venue_type == "conference"
    assert any("venue.venue_type" in c for c in paper.metadata_conflicts)


@pytest.mark.parametrize(
    "extra",
    [
        {"raw_metadata": None},
        {"publication_status": None},
        {"authors": [{"name": "Explicit Name", "affiliations": None}]},
    ],
)
def test_nullable_optional_metadata_keeps_valid_paper(extra):
    paper = run(
        BibTexMapper().to_paper(
            "@misc{x,title={Fixture}}", source="crossref", supplemental_metadata=extra
        )
    )
    assert paper.title == "Fixture"
    assert paper.institutions == []
    assert paper.source_records[0].raw_metadata["_supplemental"] == {
        k: v for k, v in extra.items() if k != "raw_metadata"
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"institutions": "Not a list"},
        {"institutions": {"name": "Not a list"}},
        {"raw_metadata": "Not a dictionary"},
    ],
)
def test_wrong_metadata_container_type_is_not_silently_accepted(extra):
    with pytest.raises(BibTexMappingError):
        run(
            BibTexMapper().to_paper(
                "@misc{x,title={Fixture}}", source="crossref", supplemental_metadata=extra
            )
        )


def test_known_publication_conflict_is_visible_and_keeps_first_value():
    left = Paper(
        title="Fixture", doi="10.1000/status", publication_status=PublicationStatus.PREPRINT
    )
    right = Paper(
        title="Fixture", doi="10.1000/status", publication_status=PublicationStatus.PUBLISHED
    )
    merged = run(PaperMerger().merge_and_deduplicate([left, right]))[0]
    assert merged.publication_status == PublicationStatus.PREPRINT
    assert any("publication_status" in c for c in merged.metadata_conflicts)


def test_merge_fills_partial_venue_fields():
    left = Paper(title="Fixture", doi="10.1000/venue", venue=Venue(name="Journal"))
    right = Paper(
        title="Fixture",
        doi="10.1000/venue",
        venue=Venue(name="Journal", venue_type="journal", volume="7"),
    )
    merged = run(PaperMerger().merge_and_deduplicate([left, right]))[0]
    assert merged.venue.venue_type == "journal"
    assert merged.venue.volume == "7"


def test_same_provider_identifier_keeps_distinct_raw_snapshots():
    first = SourceRecord(
        source="crossref", source_id="10.1000/provenance", raw_metadata={"title": "First response"}
    )
    second = first.model_copy(update={"raw_metadata": {"title": "Enriched response"}})
    left = Paper(title="Fixture", doi="10.1000/provenance", source_records=[first])
    right = Paper(title="Fixture", doi="10.1000/provenance", source_records=[second])
    merged = run(PaperMerger().merge_and_deduplicate([left, right, left]))
    assert len(merged) == 1
    assert [r.raw_metadata["title"] for r in merged[0].source_records] == [
        "First response",
        "Enriched response",
    ]
