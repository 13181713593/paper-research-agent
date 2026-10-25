import asyncio
import unittest

import httpx

from paper_agent.adapters.open_access_resolver import UnpaywallResolver
from paper_agent.domain.errors import NoOpenAccessCopyError, ProviderUnavailableError
from paper_agent.domain.models import AccessStatus, Paper

EMAIL = "student@example.edu"


def _unpaywall_transport(payload, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.unpaywall.org" in request.url.host
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


class OpenAccessResolverTest(unittest.TestCase):
    def test_arxiv_id_yields_direct_pdf_url_without_network(self):
        paper = Paper(title="An arXiv paper", arxiv_id="2401.12345v2")
        resolver = UnpaywallResolver(email="")
        resolved = asyncio.run(resolver.resolve(paper))
        self.assertIs(resolved, paper)
        self.assertEqual(str(resolved.fulltext.pdf_url), "https://arxiv.org/pdf/2401.12345")
        self.assertEqual(resolved.fulltext.access_status, AccessStatus.OPEN_ACCESS)

    def test_existing_pdf_url_is_reused(self):
        paper = Paper(title="Has a PDF URL")
        paper.fulltext.pdf_url = "https://example.org/paper.pdf"
        resolver = UnpaywallResolver(email="")
        resolved = asyncio.run(resolver.resolve(paper))
        self.assertEqual(str(resolved.fulltext.pdf_url), "https://example.org/paper.pdf")
        self.assertEqual(resolved.fulltext.access_status, AccessStatus.OPEN_ACCESS)

    def test_landing_page_oa_url_is_recorded_and_unpaywall_used(self):
        paper = Paper(title="Closed at provider", doi="10.1000/xyz", is_open_access=True)
        paper.open_access_url = "https://publisher.example.org/article/42"
        payload = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://repo.example.org/42.pdf",
                "url_for_landing_page": "https://repo.example.org/42",
                "license": "cc-by",
                "version": "acceptedVersion",
            },
        }
        resolver = UnpaywallResolver(email=EMAIL, transport=_unpaywall_transport(payload))
        resolved = asyncio.run(resolver.resolve(paper))
        self.assertEqual(str(resolved.fulltext.pdf_url), "https://repo.example.org/42.pdf")
        self.assertEqual(
            str(resolved.fulltext.landing_page_url), "https://publisher.example.org/article/42"
        )
        self.assertEqual(resolved.fulltext.license, "cc-by")
        self.assertEqual(resolved.fulltext.version, "acceptedVersion")
        self.assertEqual(resolved.fulltext.access_status, AccessStatus.OPEN_ACCESS)

    def test_unpaywall_closed_raises_with_closed_status(self):
        paper = Paper(title="Paywalled", doi="10.1000/closed")
        payload = {"is_oa": False, "best_oa_location": None, "oa_locations": []}
        resolver = UnpaywallResolver(email=EMAIL, transport=_unpaywall_transport(payload))
        with self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(resolver.resolve(paper))
        self.assertEqual(paper.fulltext.access_status, AccessStatus.CLOSED)

    def test_no_hints_raises_unavailable(self):
        paper = Paper(title="No identifiers at all")
        resolver = UnpaywallResolver(email="")
        with self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(resolver.resolve(paper))
        self.assertEqual(paper.fulltext.access_status, AccessStatus.UNAVAILABLE)

    def test_metadata_closed_flag_maps_to_closed_status(self):
        paper = Paper(title="Provider says closed", is_open_access=False)
        resolver = UnpaywallResolver(email="")
        with self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(resolver.resolve(paper))
        self.assertEqual(paper.fulltext.access_status, AccessStatus.CLOSED)

    def test_unpaywall_network_failure_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"message": "busy"})

        paper = Paper(title="Unpaywall down", doi="10.1000/busy")
        resolver = UnpaywallResolver(email=EMAIL, transport=httpx.MockTransport(handler))
        with self.assertRaises(ProviderUnavailableError):
            asyncio.run(resolver.resolve(paper))

    def test_unpaywall_doi_not_found_raises_no_copy(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "not found"})

        paper = Paper(title="Unknown DOI", doi="10.1000/missing")
        resolver = UnpaywallResolver(email=EMAIL, transport=httpx.MockTransport(handler))
        with self.assertRaises(NoOpenAccessCopyError):
            asyncio.run(resolver.resolve(paper))
        self.assertEqual(paper.fulltext.access_status, AccessStatus.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
