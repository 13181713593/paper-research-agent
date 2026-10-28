"""Cooperative deadline propagated into synchronous SDK workers by to_thread."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
from time import monotonic


@dataclass
class RequestBudget:
    deadline: float
    cancelled: Event = field(default_factory=Event)

    def remaining(self) -> float:
        remaining = self.deadline - monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise TimeoutError("Source time budget exhausted")
        return remaining


active_budget: ContextVar[RequestBudget | None] = ContextVar("retrieval_budget", default=None)


def remaining_time(default: float) -> float:
    budget = active_budget.get()
    return min(default, budget.remaining()) if budget else default


def wait_with_budget(delay: float) -> None:
    import time

    budget = active_budget.get()
    if budget:
        if delay >= budget.remaining():
            raise TimeoutError("Retry would exceed source time budget")
        if budget.cancelled.wait(delay):
            raise TimeoutError("Source request cancelled")
        budget.remaining()
    else:
        time.sleep(delay)
