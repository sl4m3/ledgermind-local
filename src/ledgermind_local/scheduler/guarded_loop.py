"""Exception-isolated polling loop for Local workers."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from .worker_state import WorkerState

logger = logging.getLogger(__name__)


class GuardedWorker(Protocol):
    def process_once(self) -> object | None: ...


class GuardedWorkerLoop:
    """Run a worker repeatedly without allowing one iteration to kill the loop."""

    def __init__(
        self,
        worker: GuardedWorker,
        *,
        state: WorkerState | None = None,
        name: str | None = None,
        poll_interval_seconds: float = 1.0,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
        close_on_stop: bool = False,
    ) -> None:
        max_backoff = max(float(max_backoff_seconds), 0.0)
        initial_backoff = min(max(float(initial_backoff_seconds), 0.0), max_backoff)
        self.worker = worker
        worker_state = getattr(worker, "state", None)
        self.state = state or (
            worker_state
            if isinstance(worker_state, WorkerState)
            else WorkerState(name or type(worker).__name__.replace(" ", "_").lower())
        )
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.0)
        self.initial_backoff_seconds = initial_backoff
        self.max_backoff_seconds = max_backoff
        self._close_on_stop = close_on_stop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._started = False

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    def request_stop(self) -> None:
        """Request a stop; the current iteration is allowed to finish."""

        self._stop.set()
        request_worker_stop = getattr(self.worker, "request_stop", None)
        if callable(request_worker_stop):
            request_worker_stop()

    stop = request_stop

    def start(self) -> threading.Thread:
        with self._thread_lock:
            if self._started:
                raise RuntimeError("worker loop has already been started")
            self._started = True
            self._thread = threading.Thread(
                target=self.run,
                name=f"ledgermind-{self.state.name}",
                daemon=True,
            )
            thread = self._thread
        thread.start()
        return thread

    def join(self, timeout: float | None = None) -> bool:
        """Join the loop thread and report whether it stopped before timeout."""

        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def run(self) -> None:
        current_thread = threading.current_thread()
        with self._thread_lock:
            if self._thread is None:
                self._thread = current_thread
            elif self._thread is not current_thread:
                raise RuntimeError("worker loop is already running in another thread")
            self._started = True

        backoff = self.initial_backoff_seconds
        try:
            while not self._stop.is_set():
                self.state.mark_started()
                try:
                    result = self.worker.process_once()
                except Exception as exc:  # noqa: BLE001 - loop boundary
                    error_code = type(exc).__name__
                    self.state.mark_error(error_code)
                    # Never include exception text: provider payloads may be sensitive.
                    logger.warning(
                        "worker loop iteration failed",
                        extra={"worker": self.state.name, "error_code": error_code},
                    )
                    if self._stop.wait(backoff):
                        break
                    if backoff > 0:
                        backoff = min(
                            self.max_backoff_seconds,
                            max(backoff * 2, self.initial_backoff_seconds),
                        )
                    continue

                item_id = _result_item_id(result)
                self.state.mark_success(
                    item_id=item_id,
                    processed=result is not None,
                )
                backoff = self.initial_backoff_seconds
                if self._stop.wait(self.poll_interval_seconds):
                    break
        finally:
            if self._close_on_stop:
                close_worker = getattr(self.worker, "close", None)
                if callable(close_worker):
                    close_worker()
            self.state.mark_stopped()


def _result_item_id(result: object | None) -> str | None:
    if result is None:
        return None
    for attribute in ("job_id", "command_id", "item_id"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = ["GuardedWorkerLoop"]
