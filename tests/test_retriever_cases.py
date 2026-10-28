import threading
from datetime import date
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import bibtexparser
import pytest
import requests

from paper_agent.adapters.arxiv_retriever import ArxivRetriever, ArxivSearchConfig
from paper_agent.domain.errors import ProviderUnavailableError
from paper_agent.domain.models import PaperSource, SearchPlan, SearchQuery


def plan(query="neural retrieval", **filters):
    return SearchPlan(
        intent_summary="test",
        queries=[SearchQuery(source=PaperSource.ARXIV, query=query, limit=2, filters=filters)],
    )


def result(**changes):
    values = dict(  # noqa: C408
        entry_id="https://arxiv.org/abs/2401.01234v1",
        title="Neural {Retrieval} & ranking",
        authors=[SimpleNamespace(name="Jane Doe")],
        summary="Abstract text.",
        published=date(2024, 1, 2),
        updated=date(2024, 1, 3),
        doi=None,
        comment=None,
        journal_ref=None,
        primary_category="cs.IR",
        categories=["cs.IR"],
        pdf_url="https://arxiv.org/pdf/2401.01234v1",
    )
    values.update(changes)
    return SimpleNamespace(**values)


class Client:
    def __init__(self, results):
        self.rows = results
        self.queries = []
        self.thread = None

    def results(self, search):
        self.thread = threading.get_ident()
        self.queries.append(search.query)
        yield from self.rows


@pytest.mark.asyncio
async def test_actual_generator_iteration_is_off_event_loop():
    c = Client([result()])
    retriever = ArxivRetriever(client=c)
    papers = await retriever.search(plan())
    assert len(papers) == 1 and c.thread != threading.get_ident()
    assert papers[0].fulltext.access_status.value == "unknown"
    assert papers[0].source_records[0].raw_metadata["title"] == "Neural {Retrieval} & ranking"


@pytest.mark.asyncio
async def test_only_own_queries_and_no_query_is_no_network():
    c = Client([])
    other = SearchPlan(
        intent_summary="test", queries=[SearchQuery(source=PaperSource.OPENALEX, query="x")]
    )
    assert await ArxivRetriever(client=c).search(other) == []
    assert c.queries == []


@pytest.mark.asyncio
async def test_empty_results_are_normal_by_default_and_error_policy_is_opt_in():
    assert await ArxivRetriever(client=Client([])).search(plan()) == []
    with pytest.raises(ProviderUnavailableError) as error:
        await ArxivRetriever(
            client=Client([]), config=ArxivSearchConfig(empty_results="error")
        ).search(plan())
    assert type(error.value).__name__ == "ArxivNoResultsError"
    r = ArxivRetriever(client=Client([]), config=ArxivSearchConfig(empty_results="return"))
    assert await r.search(plan()) == []


@pytest.mark.asyncio
async def test_empty_chinese_variant_does_not_discard_english_papers():
    class BilingualClient(Client):
        def results(self, search):
            self.queries.append(search.query)
            if "scene text" in search.query:
                yield result(title="Scene text detection")

    client = BilingualClient([])
    queries = [
        SearchQuery(source="arxiv", query=q, limit=2)
        for q in ['"scene text detection"', '"场景文本检测与识别"']
    ]
    papers = await ArxivRetriever(client=client, config=ArxivSearchConfig(delay_seconds=0)).search(
        SearchPlan(intent_summary="场景文本检测与识别", queries=queries)
    )
    assert len(client.queries) == 2
    assert len(papers) == 1 and papers[0].title == "Scene text detection"


@pytest.mark.asyncio
async def test_keyword_query_and_date_and_exclusion_are_encoded():
    c = Client([result()])
    await ArxivRetriever(client=c).search(
        plan(
            "neural retrieval",
            date_from="2024-01-01",
            date_to="2024-02-01",
            excluded_terms=["vision"],
        )
    )
    # arXiv 的 Boolean 接口会对每个 term 取 AND，多词语式极易把召回打到 0，
    # 因此只保留首个短语；其余同义表达由规划器拆成独立查询后取并集。
    assert c.queries == [
        'all:neural AND submittedDate:[202401010000 TO 202402012359] ANDNOT all:"vision"'
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,filters",
    [
        ("", {}),
        ("a", {"date_from": "bad"}),
        ("a", {"date_from": "2025-01-01", "date_to": "2024-01-01"}),
        ("a", {"unsupported": True}),
    ],
)
async def test_invalid_query_or_filter_is_explicit_error(query, filters):
    c = Client([])
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever(client=c).search(plan(query, **filters))
    assert not c.queries


@pytest.mark.asyncio
async def test_optional_missing_fields_and_invalid_dates_are_preserved():
    r = result(published="2024-02-30", summary=None, authors=[], doi="bad")
    p = (await ArxivRetriever(client=Client([r])).search(plan()))[0]
    assert p.publication_date is None and p.abstract is None and p.authors == [] and p.doi is None
    assert p.metadata_conflicts


@pytest.mark.asyncio
async def test_missing_title_is_not_silently_discarded():
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever(client=Client([result(title="")])).search(plan())


FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
<title>arXiv Query</title><id>https://export.arxiv.org/api/query</id><updated>2024-01-03T00:00:00Z</updated>
<opensearch:totalResults>1</opensearch:totalResults><opensearch:startIndex>0</opensearch:startIndex><opensearch:itemsPerPage>1</opensearch:itemsPerPage>
<entry><id>http://arxiv.org/abs/2401.01234v1</id><title>Neural retrieval</title><summary>Abstract.</summary><published>2024-01-02T00:00:00Z</published><updated>2024-01-03T00:00:00Z</updated>
<author><name>Jane Doe</name></author><arxiv:primary_category term="cs.IR"/><category term="cs.IR"/>
<link href="https://arxiv.org/pdf/2401.01234v1" type="application/pdf" title="pdf" rel="related"/></entry></feed>"""


def fake_wire(monkeypatch, replies):
    calls = []
    clock = [1000.0]
    import paper_agent.adapters.arxiv_retriever as module

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(module._RateGate, "last_started", None)
    monkeypatch.setattr(module._RateGate, "cooldown_until", 0.0, raising=False)
    monkeypatch.setattr(module.time, "time", lambda: 1700000000.0 + clock[0] - 1000.0)

    def send(self, request, **kwargs):
        calls.append({"time": clock[0], "url": request.url, "timeout": kwargs.get("timeout")})
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        status, headers, body = item
        response = requests.Response()
        response.status_code, response.headers = status, headers
        response._content = body.encode()
        response.request, response.url = request, request.url
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return calls


@pytest.mark.asyncio
async def test_real_sdk_path_429_retry_after_timeout_and_mapping(monkeypatch):
    calls = fake_wire(monkeypatch, [(429, {"Retry-After": "7"}, ""), (200, {}, FEED)])
    r = ArxivRetriever(config=ArxivSearchConfig(max_retries=1))
    papers = await r.search(plan())
    assert papers[0].abstract == "Abstract."
    assert calls[1]["time"] - calls[0]["time"] >= 7
    assert all(c["timeout"][0] > 0 and c["timeout"][1] > 0 for c in calls)
    assert parse_qs(urlsplit(calls[0]["url"]).query)["search_query"] == ["all:neural"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,name",
    [
        (requests.Timeout("wire timeout"), "ArxivTimeoutError"),
        ((429, {}, ""), "ArxivRateLimitError"),
        ((503, {}, ""), "ProviderUnavailableError"),
    ],
)
async def test_network_failures_retry_with_a_bound(monkeypatch, failure, name):
    calls = fake_wire(monkeypatch, [failure, failure])
    with pytest.raises(ProviderUnavailableError) as error:
        await ArxivRetriever(config=ArxivSearchConfig(max_retries=1)).search(plan())
    assert type(error.value).__name__ == name
    assert len(calls) == 2 and error.value.__cause__ is not None


@pytest.mark.asyncio
async def test_malformed_feed_and_missing_title_are_errors_not_success(monkeypatch):
    fake_wire(monkeypatch, [(200, {}, FEED.replace("<title>Neural retrieval</title>", ""))])
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever().search(plan())


@pytest.mark.asyncio
async def test_process_wide_rate_limit_covers_different_retrievers(monkeypatch):
    calls = fake_wire(monkeypatch, [(200, {}, FEED), (200, {}, FEED)])
    await ArxivRetriever().search(plan())
    await ArxivRetriever().search(plan())
    assert calls[1]["time"] - calls[0]["time"] >= 3


def test_real_client_cannot_disable_three_second_minimum():
    with pytest.raises(ValueError):
        ArxivRetriever(config=ArxivSearchConfig(delay_seconds=0))


@pytest.mark.asyncio
async def test_single_keyword_uses_standard_api_form_but_phrases_stay_quoted(monkeypatch):
    calls = fake_wire(monkeypatch, [(200, {}, FEED), (200, {}, FEED)])
    r = ArxivRetriever()
    await r.search(plan("electron"))
    await r.search(plan('"neural retrieval" ranking'))
    queries = [parse_qs(urlsplit(call["url"]).query)["search_query"][0] for call in calls]
    assert queries == ["all:electron", 'all:"neural retrieval"']


@pytest.mark.asyncio
async def test_sdk_does_not_request_100_rows_when_only_two_are_needed(monkeypatch):
    calls = fake_wire(monkeypatch, [(200, {}, FEED)])
    papers = await ArxivRetriever().search(plan())
    assert papers[0].title == "Neural retrieval"
    assert parse_qs(urlsplit(calls[0]["url"]).query)["max_results"] == ["2"]


@pytest.mark.asyncio
async def test_sdk_pagination_caps_last_page_at_remaining_limit(monkeypatch):
    entry = FEED[FEED.index("<entry>") : FEED.index("</entry>") + len("</entry>")]
    first_page = (
        FEED.replace("totalResults>1", "totalResults>3")
        .replace("itemsPerPage>1", "itemsPerPage>2")
        .replace(entry, entry + entry.replace("2401.01234v1", "2401.01235v1"))
    )
    last_page = (
        FEED.replace("totalResults>1", "totalResults>3")
        .replace("startIndex>0", "startIndex>2")
        .replace("2401.01234v1", "2401.01236v1")
    )
    calls = fake_wire(monkeypatch, [(200, {}, first_page), (200, {}, last_page)])
    search_plan = plan()
    search_plan.queries[0].limit = 3
    papers = await ArxivRetriever(config=ArxivSearchConfig(page_size=2)).search(search_plan)
    assert [paper.arxiv_id for paper in papers] == [
        "2401.01234v1",
        "2401.01235v1",
        "2401.01236v1",
    ]
    parameters = [parse_qs(urlsplit(call["url"]).query) for call in calls]
    assert [(p["start"][0], p["max_results"][0]) for p in parameters] == [("0", "2"), ("2", "1")]


@pytest.mark.asyncio
async def test_exported_arxiv_bibtex_contains_available_citation_fields():
    r = result(title="Neural Retrieval & $x^2$", doi="https://doi.org/10.1234/ABC")
    retriever = ArxivRetriever(client=Client([r]), config=ArxivSearchConfig(delay_seconds=0))
    paper = (await retriever.search(plan()))[0]
    entry = bibtexparser.loads(paper.bibtex).entries[0]
    assert entry["author"] == "Jane Doe"
    assert entry["year"] == "2024"
    assert entry["eprint"] == "2401.01234v1"
    assert entry["archiveprefix"] == "arXiv"
    assert entry["primaryclass"] == "cs.IR"
    assert entry["doi"] == "10.1234/abc"
    assert entry["url"] == r.entry_id
    assert entry["ID"] != "arxiv_entry"
    assert r"\&" in entry["title"] and "$x^2$" in entry["title"]
    assert paper.source_records[0].raw_metadata["_bibtex"] == paper.bibtex
    from paper_agent.adapters import BibTexMapper

    restored = await BibTexMapper().to_paper(paper.bibtex, source="arxiv")
    assert restored.title == paper.title
    assert [a.name for a in restored.authors] == ["Jane Doe"]
    assert restored.year == 2024 and restored.arxiv_id == paper.arxiv_id
    assert restored.doi == paper.doi


@pytest.mark.asyncio
async def test_export_does_not_invent_missing_or_invalid_citation_fields():
    r = result(published="2024-02-30", authors=[], doi="invalid", primary_category=None)
    retriever = ArxivRetriever(client=Client([r]), config=ArxivSearchConfig(delay_seconds=0))
    paper = (await retriever.search(plan()))[0]
    entry = bibtexparser.loads(paper.bibtex).entries[0]
    assert entry["eprint"] == "2401.01234v1"
    assert not {"year", "author", "doi", "primaryclass"} & entry.keys()
    assert paper.metadata_conflicts


@pytest.mark.asyncio
async def test_injected_sdk_generator_is_not_consumed_past_query_limit():
    class OverflowClient:
        def results(self, search):
            yield result()
            raise AssertionError("Iterator consumed beyond the requested one result")

    p = plan()
    p.queries[0].limit = 1
    r = ArxivRetriever(
        client=OverflowClient(), config=ArxivSearchConfig(delay_seconds=0, max_retries=0)
    )
    assert len(await r.search(p)) == 1


@pytest.mark.asyncio
async def test_429_without_server_hint_waits_for_recovery_and_increases_backoff(monkeypatch):
    calls = fake_wire(monkeypatch, [(429, {}, ""), (429, {}, ""), (200, {}, FEED)])
    papers = await ArxivRetriever(config=ArxivSearchConfig(max_retries=2)).search(plan())
    assert papers[0].title == "Neural retrieval"
    assert len(calls) == 3
    assert calls[1]["time"] - calls[0]["time"] >= 30
    assert calls[2]["time"] - calls[1]["time"] >= 60


@pytest.mark.asyncio
async def test_429_cooldown_applies_to_next_instance_even_when_no_retry_is_requested(monkeypatch):
    calls = fake_wire(monkeypatch, [(429, {}, ""), (200, {}, FEED)])
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever(config=ArxivSearchConfig(max_retries=0)).search(plan())
    papers = await ArxivRetriever(config=ArxivSearchConfig(max_retries=0)).search(plan())
    assert papers[0].title == "Neural retrieval"
    assert calls[1]["time"] - calls[0]["time"] >= 30


@pytest.mark.asyncio
async def test_retry_after_http_date_respects_server_clock(monkeypatch):
    calls = fake_wire(
        monkeypatch,
        [
            (
                429,
                {
                    "Date": "Tue, 14 Nov 2023 22:13:20 GMT",
                    "Retry-After": "Tue, 14 Nov 2023 22:14:20 GMT",
                },
                "",
            ),
            (200, {}, FEED),
        ],
    )
    await ArxivRetriever(config=ArxivSearchConfig(max_retries=1)).search(plan())
    assert calls[1]["time"] - calls[0]["time"] >= 60


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["nan", "inf", "-inf", "-1", "not-a-date"])
async def test_invalid_retry_after_uses_safe_recovery_wait(monkeypatch, hint):
    calls = fake_wire(monkeypatch, [(429, {"Retry-After": hint}, ""), (200, {}, FEED)])
    papers = await ArxivRetriever(config=ArxivSearchConfig(max_retries=1)).search(plan())
    assert papers[0].title == "Neural retrieval"
    assert calls[1]["time"] - calls[0]["time"] >= 30


@pytest.mark.asyncio
async def test_timeout_does_not_reuse_retry_header_from_an_earlier_response(monkeypatch):
    calls = fake_wire(
        monkeypatch,
        [
            (429, {"Retry-After": "10"}, ""),
            requests.Timeout("next response never arrived"),
            (200, {}, FEED),
        ],
    )
    papers = await ArxivRetriever(config=ArxivSearchConfig(max_retries=2)).search(plan())
    assert papers[0].title == "Neural retrieval"
    assert calls[1]["time"] - calls[0]["time"] >= 10
    assert 3 <= calls[2]["time"] - calls[1]["time"] < 10


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_permanent_http_failure_does_not_send_pointless_retries(monkeypatch, status):
    calls = fake_wire(monkeypatch, [(status, {}, "")] * 3)
    with pytest.raises(ProviderUnavailableError, match=str(status)):
        await ArxivRetriever(config=ArxivSearchConfig(max_retries=2)).search(plan())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_503_retry_after_is_honored(monkeypatch):
    calls = fake_wire(monkeypatch, [(503, {"Retry-After": "12"}, ""), (200, {}, FEED)])
    assert await ArxivRetriever(config=ArxivSearchConfig(max_retries=1)).search(plan())
    assert calls[1]["time"] - calls[0]["time"] >= 12


@pytest.mark.asyncio
async def test_long_server_cooldown_fails_promptly_and_never_retries_early(monkeypatch):
    calls = fake_wire(monkeypatch, [(429, {"Retry-After": "600"}, "")] * 3)
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever(config=ArxivSearchConfig(max_retries=2)).search(plan())
    assert len(calls) == 1
    # A different instance must not bypass that service-wide cooldown.
    with pytest.raises(ProviderUnavailableError):
        await ArxivRetriever(config=ArxivSearchConfig(max_retries=0)).search(plan())
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name,value", [("delay_seconds", float("nan")), ("timeout_read", float("inf"))]
)
def test_non_finite_wait_policy_is_rejected(name, value):
    with pytest.raises(ValueError):
        ArxivSearchConfig(**{name: value})
