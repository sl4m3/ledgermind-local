"""Projection outbox worker."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ledgermind_local.persistence import SQLiteUnitOfWork
from ledgermind_local.projections import _ProjectionHandler
from ledgermind_local.projections.dispatcher import ProjectionDispatcher

_RETRY_DELAYS_SECONDS = (1, 5, 30, 300, 1800)


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.isoformat()


def _retry_delay_seconds(attempt: int) -> int:
    if attempt <= 1:
        return _RETRY_DELAYS_SECONDS[0]
    if attempt >= len(_RETRY_DELAYS_SECONDS):
        return _RETRY_DELAYS_SECONDS[-1]
    return _RETRY_DELAYS_SECONDS[attempt - 1]


class OutboxWorker:
    """Worker that drains projection deliveries from durable outbox."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        dispatcher: ProjectionDispatcher,
        projection_handlers_factory: Callable[..., Mapping[str, _ProjectionHandler]] | None = None,
        worker_id: str,
        poll_interval_seconds: float = 1.0,
        stale_claim_ttl_seconds: int = 30,
        now_factory: Callable[[], datetime] | None = None,
        sleep: Callable[..., Any] | None = None,
        unit_of_work_factory: Callable[..., Any] | None = None,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self._database_path = str(database_path)
        self._dispatcher = dispatcher
        self._worker_id = worker_id
        self._poll_interval = poll_interval_seconds
        self._stale_claim_ttl = stale_claim_ttl_seconds
        self._now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        if sleep is None:
            sleep = threading.Event().wait
        self._sleep = sleep
        self._unit_of_work_factory = (
            unit_of_work_factory or (lambda: SQLiteUnitOfWork(self._database_path))
        )
        self._close_callback = close_callback
        self._closed = False
        if projection_handlers_factory is not None:
            handlers = projection_handlers_factory()
            self._dispatcher = ProjectionDispatcher(handlers)
            self._owned_handlers: Mapping[str, _ProjectionHandler] | None = handlers
        else:
            self._owned_handlers = None
        self._stop_requested = False

    @property
    def projection_names(self) -> tuple[str, ...]:
        return self._dispatcher.projection_names

    def request_stop(self) -> None:
        self._stop_requested = True

    def close(self) -> None:
        """Release long-lived projection handlers and their backing resources."""

        if self._closed:
            return
        self._closed = True
        if self._owned_handlers is not None:
            self._close_handlers(self._owned_handlers)
        if self._close_callback is not None:
            self._close_callback()

    def run(self) -> None:
        while not self._stop_requested:
            processed = self.run_once()
            if not processed:
                self._interruptible_sleep(self._poll_interval)

    def run_once(self) -> bool:
        processed_any = False
        for projection_name in self.projection_names:
            if self._stop_requested:
                break

            while not self._stop_requested:
                if not self._process_projection_once(projection_name):
                    break
                processed_any = True
        return processed_any

    def _process_projection_once(self, projection_name: str) -> bool:
        now = self._now_factory()
        stale_cutoff = now - timedelta(seconds=self._stale_claim_ttl)

        event = self._claim_projection(
            projection_name=projection_name,
            now=now,
            stale_cutoff=stale_cutoff,
        )
        if event is None:
            return False

        try:
            self._dispatcher.dispatch(projection_name, event)
            self._mark_processed(
                projection_name=projection_name,
                event=event,
                processed_at=now,
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_failed(
                projection_name=projection_name,
                event=event,
                now=now,
                error=exc,
            )
        return True

    def _claim_projection(
        self,
        *,
        projection_name: str,
        now: datetime,
        stale_cutoff: datetime,
    ) -> Any:
        with self._unit_of_work_factory() as uow:
            event = uow.outbox.acquire_next(
                projection_name=projection_name,
                worker_id=self._worker_id,
                now=_to_iso(now),
                stale_claim_before=_to_iso(stale_cutoff),
            )
            if event is None:
                return None
            uow.commit()
            return event

    def _mark_processed(
        self,
        *,
        projection_name: str,
        event: Any,
        processed_at: datetime,
    ) -> bool:
        with self._unit_of_work_factory() as uow:
            marked = uow.outbox.mark_processed(
                projection_name,
                event.event_id,
                processed_at=_to_iso(processed_at),
                claimed_by=self._worker_id,
            )
            if not marked:
                uow.rollback()
                return False
            uow.commit()
            return True

    def _mark_failed(
        self,
        *,
        projection_name: str,
        event: Any,
        now: datetime,
        error: Exception,
    ) -> int:
        with self._unit_of_work_factory() as uow:
            attempts = uow.outbox.mark_failed(
                projection_name,
                event.event_id,
                available_at=_to_iso(
                    now + timedelta(seconds=_retry_delay_seconds(event.attempts + 1))
                ),
                last_error=f"{type(error).__name__}: {error}",
                claimed_by=self._worker_id,
            )
            if attempts == 0:
                uow.rollback()
                return 0
            uow.commit()
            return attempts

    @staticmethod
    def _close_handlers(handlers: Mapping[str, _ProjectionHandler]) -> None:
        for handler in handlers.values():
            close = getattr(handler, "close", None)
            if callable(close):
                close()

    def _interruptible_sleep(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 0 and not self._stop_requested:
            delay = min(remaining, 0.05)
            self._sleep(delay)
            remaining -= delay
