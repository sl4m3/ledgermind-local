"""Durable worker that retries extraction after Hermes state becomes readable."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .spool import FileSpool


class PendingExtractionNotReady(RuntimeError):
    """The source round is not available in state.db yet."""


@dataclass(frozen=True, slots=True)
class PendingExtractionResult:
    request: Mapping[str, Any]


PendingProcessor = Callable[[Mapping[str, Any]], PendingExtractionResult | None]


class PendingExtractionWorker:
    """Process durable pending extraction envelopes with bounded retries."""

    def __init__(
        self,
        *,
        spool: FileSpool,
        batch_size: int = 2,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
        max_attempts: int = 12,
    ) -> None:
        self._spool = spool
        self._batch_size = max(int(batch_size), 1)
        self._base_backoff_seconds = max(float(base_backoff_seconds), 0.1)
        self._max_backoff_seconds = max(float(max_backoff_seconds), self._base_backoff_seconds)
        self._max_attempts = max(int(max_attempts), 1)
        self._processor: PendingProcessor | None = None
        self._processor_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None

    def set_processor(self, processor: PendingProcessor | None) -> None:
        with self._processor_lock:
            self._processor = processor
        self.wake()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="ledgermind-hermes-pending-extraction",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def wake(self) -> None:
        self._wakeup.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            has_work = self._drain_once()
            timeout = 0.1 if has_work else self._base_backoff_seconds
            self._wakeup.wait(timeout=timeout)
            self._wakeup.clear()

    def _drain_once(self) -> bool:
        with self._processor_lock:
            processor = self._processor
        if processor is None:
            return False

        items = self._spool.pop_pending(limit=self._batch_size)
        for item_name, payload in items:
            self._process_item(item_name=item_name, payload=payload, processor=processor)
        return bool(items)

    def _process_item(
        self,
        *,
        item_name: str,
        payload: Mapping[str, Any],
        processor: PendingProcessor,
    ) -> None:
        try:
            result = processor(payload)
            if result is None:
                self._spool.complete_pending_item(item_name)
                return
            request = result.request
            idempotency_key = request.get("idempotency_key")
            if not isinstance(idempotency_key, str) or not idempotency_key:
                raise ValueError("pending processor returned request without idempotency_key")
            self._spool.enqueue_ready(idempotency_key, request)
            self._spool.complete_pending_item(item_name)
        except PendingExtractionNotReady as exc:
            self._retry(item_name=item_name, payload=payload, reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            self._retry(item_name=item_name, payload=payload, reason=f"{type(exc).__name__}: {exc}")

    def _retry(self, *, item_name: str, payload: Mapping[str, Any], reason: str) -> None:
        updated = dict(payload)
        attempts = int(updated.get("attempts", 0)) + 1
        updated["attempts"] = attempts
        updated["last_error"] = reason
        if attempts >= self._max_attempts:
            self._spool.fail_pending_item(item_name, reason=f"max_attempts_exceeded:{reason}")
            return
        delay = min(
            self._max_backoff_seconds,
            self._base_backoff_seconds * (2 ** max(attempts - 1, 0)),
        )
        updated["next_attempt_at"] = time.time() + delay
        self._spool.replace_pending(item_name, updated)
