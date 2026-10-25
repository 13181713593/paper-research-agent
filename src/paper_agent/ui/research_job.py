"""Run research independently of Streamlit reruns; expose thread-safe snapshots."""

from __future__ import annotations

import asyncio
from threading import Lock, Thread

from paper_agent.domain.models import PipelineResult, StageEvent


class ResearchJob:
    def __init__(self, agent, request):
        self.request = request
        self._agent = agent
        self._lock = Lock()
        self._events: dict[tuple, StageEvent] = {}
        self._result: PipelineResult | None = None
        self._error: str | None = None
        self._done = False
        self._thread = Thread(target=self._run, daemon=True, name="paper-research")

    def start(self):
        self._thread.start()
        # Small offline/demo runs can finish without a polling cycle.
        self._thread.join(timeout=0.15)
        return self

    async def _record(self, event):
        key = (event.stage, event.scope, event.source_name, event.paper_id)
        with self._lock:
            self._events[key] = event.model_copy(deep=True)

    def _run(self):
        try:
            result = asyncio.run(self._agent.research(self.request, event_sink=self._record))
            with self._lock:
                self._result = result
        except Exception as exc:  # noqa: BLE001 - preserve UI history on worker failure
            with self._lock:
                self._error = f"研究任务未完成（{type(exc).__name__}），已产生的阶段结果仍可查看。"
        finally:
            with self._lock:
                self._done = True

    def snapshot(self):
        with self._lock:
            return list(self._events.values()), self._result, self._error, self._done
