"""Background loop for durable RawRound processing."""

from __future__ import annotations

import threading
from typing import Protocol


class _ProcessingWorker(Protocol):
    def process_once(self) -> object | None: ...


class ProcessingWorkerLoop:
    def __init__(
        self, worker: _ProcessingWorker, *, poll_interval_seconds: float = 1.0
    ) -> None:
        self.worker = worker
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0)
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            self.worker.process_once()
            self._stop.wait(self.poll_interval_seconds)
