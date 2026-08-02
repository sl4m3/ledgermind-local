"""Background worker for Hermes outbox delivery."""

from __future__ import annotations

import threading
from typing import Any, Mapping

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
    ) -> None:
        self._spool = spool
        self._client = client
        self._batch_size = max(int(batch_size), 1)
        self._request_timeout = max(float(request_timeout), 0.1)
        self._base_backoff_seconds = max(float(base_backoff_seconds), 0.1)
        self._max_backoff_seconds = max(float(max_backoff_seconds), self._base_backoff_seconds)
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
        if self._client is None:
            payload = dict(payload)
            payload["delivery"] = delivery
            self._spool.mark_retry(item_name, payload)
            return

        payload = dict(payload)
        payload["delivery"] = delivery

        try:
            self._client.ingest_atom(payload, timeout=self._request_timeout)
            self._spool.complete_item(item_name)
            return
        except LedgerMindConflictError as exc:
            self._spool.mark_failed(item_name, reason=f"idempotency_conflict:{exc}")
            return
        except LedgerMindUnauthorizedError as exc:
            self._spool.mark_retry(item_name, payload)
            return
        except (LedgerMindNetworkError,):
            self._spool.mark_retry(item_name, payload)
            return
        except LedgerMindResponseError as exc:
            self._spool.mark_failed(item_name, reason=f"invalid_payload:{exc}")
            return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._spool.mark_failed(item_name, reason=f"delivery_error:{exc}")
            return
