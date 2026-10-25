import copy
from datetime import date

import pytest

from paper_agent.adapters.bibtex_mapper import BibTexMapper, BibTexMappingError
from paper_agent.domain.models import AccessStatus, PublicationStatus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bib",
    [
        "",
        "not bibtex",
        "@article{x,title={broken}",
        "@article{x,year={2024}}",
        "@article{x,title={ }}",
    ],
)
async def test_unusable_bibtex_is_a_typed_error(bib):
    with pytest.raises(BibTexMappingError):
        await BibTexMapper().to_paper(bib, source="arxiv")


@pytest.mark.asyncio
async def test_corporate_author_and_latex_accents_and_symbols():
    p = await BibTexMapper().to_paper(
        r"@misc{x,title={{BERT} for $x^2$ \& search},author={{Research and Development Team} and M{\"u}ller, Anna}}",
        source="crossref",
    )
    assert p.title == "BERT for $x^2$ & search"
    assert [a.name for a in p.authors] == ["Research and Development Team", "Anna Müller"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["2024-02-30", "not-a-date", "2024-13-01"])
async def test_invalid_date_retains_original_and_diagnostic(invalid):
    p = await BibTexMapper().to_paper(
        "@article{x,title={Test},year={2024}}",
        source="arxiv",
        supplemental_metadata={"published": invalid},
    )
    assert p.publication_date is None and p.year == 2024
    assert p.metadata_conflicts
    assert p.source_records[0].raw_metadata["_supplemental"]["published"] == invalid


@pytest.mark.asyncio
async def test_only_year_is_not_fake_full_date():
    p = await BibTexMapper().to_paper("@article{x,title={Test},year={2024}}", source="crossref")
    assert p.year == 2024 and p.publication_date is None
    assert p.publication_status == PublicationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_full_date_wins_year_conflict_and_keeps_evidence():
    p = await BibTexMapper().to_paper(
        "@article{x,title={Test},year={2023}}",
        source="arxiv",
        supplemental_metadata={"published": "2024-01-02"},
    )
    assert p.year == 2024 and p.publication_date == date(2024, 1, 2)
    assert any("year" in s for s in p.metadata_conflicts)


@pytest.mark.asyncio
async def test_metadata_is_not_mutated_and_bad_optional_urls_are_diagnosed():
    extra = {
        "abstract": "  real abstract  ",
        "pdf_url": "not-a-url",
        "raw_metadata": {"items": [1, 2]},
    }
    before = copy.deepcopy(extra)
    p = await BibTexMapper().to_paper(
        "@misc{x,title={T}}", source="arxiv", supplemental_metadata=extra
    )
    assert extra == before
    assert p.abstract == "real abstract"
    assert p.fulltext.pdf_url is None
    assert p.fulltext.access_status == AccessStatus.UNKNOWN
    assert any("pdf_url" in s for s in p.metadata_conflicts)


@pytest.mark.asyncio
async def test_affiliations_require_explicit_author_evidence():
    p = await BibTexMapper().to_paper(
        "@misc{x,title={T},author={Jane Doe}}",
        source="arxiv",
        supplemental_metadata={
            "authors": [{"name": "Jane Doe", "affiliations": ["Actual University"]}],
            "institutions": ["Actual University"],
        },
    )
    assert p.authors[0].affiliations == ["Actual University"]
    q = await BibTexMapper().to_paper("@misc{x,title={T},author={Jane Doe}}", source="arxiv")
    assert q.authors[0].affiliations == [] and q.institutions == []


@pytest.mark.asyncio
async def test_unknown_provider_is_not_mislabeled():
    with pytest.raises(BibTexMappingError):
        await BibTexMapper().to_paper("@misc{x,title={T}}", source="typo_provider")


@pytest.mark.asyncio
async def test_bibtex_abstract_used_and_raw_entry_preserved():
    bib = "@misc{x,title={T},abstract={Provider abstract},eprint={hep-th/9901001v3},archivePrefix={arXiv}}"
    p = await BibTexMapper().to_paper(bib, source="arxiv")
    assert p.abstract == "Provider abstract" and p.bibtex == bib
    assert p.arxiv_id == "hep-th/9901001v3"
    assert p.fulltext.version == "v3"
    assert p.source_records[0].raw_metadata["_bibtex"] == bib


@pytest.mark.asyncio
async def test_invalid_doi_and_year_are_not_canonical_values():
    p = await BibTexMapper().to_paper(
        "@misc{x,title={T},year={3024},doi={no DOI}}", source="crossref"
    )
    assert p.doi is None and p.year is None
    assert len(p.metadata_conflicts) >= 2


@pytest.mark.asyncio
async def test_nonserializable_provider_payload_is_rejected():
    with pytest.raises(BibTexMappingError):
        await BibTexMapper().to_paper(
            "@misc{x,title={T}}",
            source="arxiv",
            supplemental_metadata={"raw_metadata": {"object": object()}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("month", ["June", "January", "September"])
async def test_crossref_full_month_macro_with_json_title_supplement(month):
    # Real Crossref transform pattern: a full month macro and no BibTeX title.
    bib = f"@misc{{2026, DOI={{10.1002/9781394374717.ch03}}, year={{2026}}, month={month}" + "}"
    paper = await BibTexMapper().to_paper(
        bib,
        source="crossref",
        supplemental_metadata={"title": "Retrieval-Augmented Generation"},
    )
    assert paper.title == "Retrieval-Augmented Generation"
    assert paper.doi == "10.1002/9781394374717.ch03"
    assert paper.year == 2026 and paper.publication_date is None
    assert paper.authors == [] and paper.abstract is None
    assert paper.bibtex == bib
    assert paper.source_records[0].raw_metadata["_bibtex"] == bib


@pytest.mark.asyncio
async def test_explicit_string_definition_overrides_month_alias():
    paper = await BibTexMapper().to_paper(
        "@string{june={Explicit provider title}} @misc{x,title=June}",
        source="crossref",
    )
    assert paper.title == "Explicit provider title"


@pytest.mark.asyncio
async def test_undefined_non_month_macro_remains_a_typed_error():
    with pytest.raises(BibTexMappingError):
        await BibTexMapper().to_paper("@misc{x,title=missing_macro}", source="crossref")
