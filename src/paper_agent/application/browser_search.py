"""A UI-independent browser search job with immutable cross-thread snapshots."""

import asyncio
from threading import Lock, Thread


class BrowserSearchJob:
    def __init__(self, service, queries, engines, limit=10):
        self._service = service
        self.queries = list(queries)
        self.engines = tuple(engines)
        self.limit = limit
        self._lock = Lock()
        self._result = None
        self._error = None
        self._done = False
        self._thread = Thread(target=self._run, daemon=True, name="browser-paper-search")

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        try:
            result = asyncio.run(
                self._service.search(self.queries, engines=self.engines, limit=self.limit)
            )
            with self._lock:
                self._result = result
        except Exception as exc:  # noqa: BLE001 - job boundary
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._done = True

    def snapshot(self):
        with self._lock:
            return (
                self._result.model_copy(deep=True) if self._result else None,
                self._error,
                self._done,
            )
