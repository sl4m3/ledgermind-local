"""Background worker for Hermes outbox delivery."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from .client import (
    LedgerMindConflictError,
    LedgerMindNetworkError,
    LedgerMindResponseError,
    LedgerMindUnauthorizedError,
)
from .spool import FileSpool


class DeliveryWorker:
    def __init__(
        self,
        spool: FileSpool,
        client: Any,
        *,
        batch_size: int = 10,
        request_timeout: float = 5.0,
        base_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 5.0,
        max_attempts: int = 8,
    ) -> None:
        self._spool = spool
        self._client = client
        self._batch_size = max(int(batch_size), 1)
        self._request_timeout = max(float(request_timeout), 0.1)
        self._base_backoff_seconds = max(float(base_backoff_seconds), 0.1)
        self._max_backoff_seconds = max(float(max_backoff_seconds), self._base_backoff_seconds)
        self._max_attempts = max(int(max_attempts), 1)
        self._stop_event = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._backoff_seconds = self._base_backoff_seconds

    def start(self) -> None:
        if self._thread is not None or self._client is None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="ledgermind-hermes-delivery",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def wake(self) -> None:
        self._wakeup.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            has_work = self._drain_once()
            if has_work:
                self._backoff_seconds = self._base_backoff_seconds
            else:
                self._backoff_seconds = min(
                    self._max_backoff_seconds,
                    self._backoff_seconds * 2,
                )

            self._wakeup.wait(timeout=self._backoff_seconds)
            self._wakeup.clear()

    def _drain_once(self) -> bool:
        has_work = False
        for item_name, payload in self._spool.pop_ready(limit=self._batch_size):
            has_work = True
            self._process_item(item_name=item_name, payload=payload)
        return has_work

    def _process_item(self, *, item_name: str, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, dict):
            self._spool.mark_failed(item_name, reason="invalid_payload")
            return

        delivery = dict(payload.get("delivery", {}))
        delivery["attempts"] = int(delivery.get("attempts", 0)) + 1
        request = payload.get("request")
        if not isinstance(request, Mapping):
            request = dict(payload)
            request.pop("delivery", None)
            if not request:
                self._spool.mark_failed(item_name, reason="invalid_request_envelope")
                return
        payload = dict(payload)
        payload["delivery"] = delivery
        if delivery["attempts"] > self._max_attempts:
            self._spool.mark_failed(item_name, reason="max_attempts_exceeded")
            return
        if self._client is None:
            self._set_retry_time(delivery)
            self._spool.mark_retry(item_name, payload)
            return

        try:
            self._client.ingest_atom(dict(request), timeout=self._request_timeout)
            self._spool.complete_item(item_name)
            return
        except LedgerMindConflictError as exc:
            self._spool.mark_failed(item_name, reason=f"idempotency_conflict:{exc}")
            return
        except LedgerMindUnauthorizedError:
            self._set_retry_time(delivery)
            self._spool.mark_retry(item_name, payload)
            return
        except LedgerMindNetworkError:
            self._set_retry_time(delivery)
            self._spool.mark_retry(item_name, payload)
            return
        except LedgerMindResponseError as exc:
            self._spool.mark_failed(item_name, reason=f"invalid_payload:{exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self._spool.mark_failed(item_name, reason=f"delivery_error:{exc}")
            return

    def _set_retry_time(self, delivery: dict[str, Any]) -> None:
        attempts = max(int(delivery.get("attempts", 1)), 1)
        delay = min(
            self._max_backoff_seconds,
            self._base_backoff_seconds * (2 ** max(attempts - 1, 0)),
        )
        delivery["next_attempt_at"] = time.time() + delay
