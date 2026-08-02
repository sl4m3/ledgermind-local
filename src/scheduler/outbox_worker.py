"""Projection outbox worker."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from persistence import SQLiteUnitOfWork
from projections import _ProjectionHandler
from projections.dispatcher import ProjectionDispatcher

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
        self._projection_handlers_factory = projection_handlers_factory
        self._stop_requested = False

    @property
    def projection_names(self) -> tuple[str, ...]:
        return self._dispatcher.projection_names

    def request_stop(self) -> None:
        self._stop_requested = True

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

        with self._unit_of_work_factory() as uow:
            dispatcher = (
                self._dispatcher
                if self._projection_handlers_factory is None
                else ProjectionDispatcher(self._projection_handlers_factory(uow))
            )
            try:
                event = uow.outbox.acquire_next(
                    projection_name=projection_name,
                    worker_id=self._worker_id,
                    now=_to_iso(now),
                    stale_claim_before=_to_iso(stale_cutoff),
                )
                if event is None:
                    return False

                try:
                    dispatcher.dispatch(projection_name, event)
                except Exception as exc:  # noqa: BLE001
                    attempts = uow.outbox.mark_failed(
                        projection_name,
                        event.event_id,
                        available_at=_to_iso(
                            now + timedelta(seconds=_retry_delay_seconds(event.attempts + 1))
                        ),
                        last_error=f"{type(exc).__name__}: {exc}",
                    )
                    # Keep behavior safe if mark_failed could not find row due race.
                    if attempts == 0:
                        uow.rollback()
                        return False
                else:
                    uow.outbox.mark_processed(
                        projection_name,
                        event.event_id,
                        processed_at=_to_iso(now),
                    )

                uow.commit()
                return True
            except Exception:
                uow.rollback()
                raise

    def _interruptible_sleep(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 0 and not self._stop_requested:
            delay = min(remaining, 0.05)
            self._sleep(delay)
            remaining -= delay
