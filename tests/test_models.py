import unittest
from datetime import date

from pydantic import ValidationError

from paper_agent.bootstrap import build_agent
from paper_agent.domain.models import CodeAvailability, Paper


class DomainModelTest(unittest.TestCase):
    def test_minimal_paper_uses_explicit_unknown_states(self):
        paper = Paper(title="A minimally described paper")
        self.assertIsNone(paper.abstract)
        self.assertIsNone(paper.is_open_access)
        self.assertEqual(paper.authors, [])
        self.assertEqual(paper.code.status, CodeAvailability.UNKNOWN)

    def test_publication_date_populates_year(self):
        paper = Paper(title="Dated paper", publication_date=date(2026, 9, 21))
        self.assertEqual(paper.year, 2026)

    def test_conflicting_year_is_rejected(self):
        with self.assertRaises(ValidationError):
            Paper(title="Bad date", publication_date=date(2026, 9, 21), year=2025)

    def test_main_agent_contract_has_required_order(self):
        stages = [stage.value for stage in build_agent().contract().stages]
        self.assertLess(stages.index("retrieved"), stages.index("deduped"))
        self.assertLess(stages.index("pre_ranked"), stages.index("fulltext_processed"))
        self.assertLess(stages.index("verified"), stages.index("final_ranked"))


if __name__ == "__main__":
    unittest.main()
