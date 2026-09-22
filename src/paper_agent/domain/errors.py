"""Typed errors crossing adapter/application boundaries."""


class PaperAgentError(Exception):
    """Base class for expected project errors."""


class ProviderUnavailableError(PaperAgentError):
    """A metadata provider timed out, rate-limited, or rejected the request."""


class NoOpenAccessCopyError(PaperAgentError):
    """No legally downloadable open-access copy could be verified."""


class FullTextParseError(PaperAgentError):
    """The downloaded file could not be converted to usable Markdown."""


class RankingError(PaperAgentError):
    """A ranker failed or returned invalid/non-comparable scores."""


class SummaryValidationError(PaperAgentError):
    """A generated summary failed schema or evidence validation."""

