"""Background worker for Core-owned projection event delivery."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.projection_consumer import (
    CoreProjectionConsumer,
    ProjectionConsumerStats,
)
from ledgermind_local.core_gateway.projection_inbox import CoreProjectionInbox
from ledgermind_local.persistence import open_sqlite_connection

from .guarded_loop import GuardedWorkerLoop
from .worker_state import WorkerState


@dataclass(frozen=True, slots=True)
class CoreProjectionWorkerStats:
    fetched: int = 0
    persisted: int = 0
    acknowledged: int = 0
    processed: int = 0
    failed: int = 0


HandlerFactory = Callable[[sqlite3.Connection], Mapping[str, Any]]


class CoreProjectionWorker:
    """Poll Core events for Local memory spaces and apply Local projections."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        gateway: CoreGateway,
        consumer_id: str,
        handlers_factory: HandlerFactory,
        poll_limit: int = 100,
        state: WorkerState | None = None,
    ) -> None:
        if not consumer_id.strip():
            raise ValueError("consumer_id must not be empty")
        if not 1 <= poll_limit <= 1000:
            raise ValueError("poll_limit must be between 1 and 1000")
        self._database_path = str(database_path)
        self._gateway = gateway
        self._consumer_id = consumer_id
        self._handlers_factory = handlers_factory
        self._poll_limit = poll_limit
        self.state = state or WorkerState("core-projection")
        self._stop = threading.Event()
        self._connection: sqlite3.Connection | None = None
        self._consumers: dict[str, CoreProjectionConsumer] | None = None
        self._handlers: Mapping[str, Any] | None = None
        self._closed = False
        self._loop: GuardedWorkerLoop | None = None

    def process_once(self) -> CoreProjectionWorkerStats:
        if self._closed:
            raise RuntimeError("Core projection worker is closed")
        if self._stop.is_set():
            return CoreProjectionWorkerStats()
        consumers = self._ensure_consumers()
        assert self._connection is not None

        totals = CoreProjectionWorkerStats()
        rows = self._connection.execute(
            """
            SELECT memory_space_id
            FROM memory_spaces
            ORDER BY memory_space_id ASC
            """
        ).fetchall()
        for projection_name, consumer in consumers.items():
            if self._stop.is_set():
                break
            for row in rows:
                if self._stop.is_set():
                    break
                result = consumer.poll_once(
                    str(row[0]),
                    consumer_id=f"{self._consumer_id}:{projection_name}",
                    limit=self._poll_limit,
                )
                totals = _add_stats(totals, result)
        return totals

    def request_stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.stop_event.set()

    def create_loop(
        self,
        *,
        poll_interval_seconds: float = 1.0,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
    ) -> GuardedWorkerLoop:
        loop = GuardedWorkerLoop(
            self,
            state=self.state,
            name="core-projection",
            poll_interval_seconds=poll_interval_seconds,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            close_on_stop=True,
        )
        self._loop = loop
        if self._stop.is_set():
            loop.stop_event.set()
        return loop

    def run_loop(
        self,
        *,
        poll_interval_seconds: float = 1.0,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.create_loop(
            poll_interval_seconds=poll_interval_seconds,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        ).run()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handlers is not None:
            for handler in self._handlers.values():
                close = getattr(handler, "close", None)
                if callable(close):
                    close()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _ensure_consumers(self) -> dict[str, CoreProjectionConsumer]:
        if self._consumers is not None:
            return self._consumers
        connection = open_sqlite_connection(self._database_path)
        try:
            handlers = dict(self._handlers_factory(connection))
            if not handlers:
                raise ValueError("at least one Core projection handler is required")
            inbox = CoreProjectionInbox(connection)
            consumers = {
                name: CoreProjectionConsumer(
                    gateway=self._gateway,
                    inbox=inbox,
                    handlers={name: handler},
                )
                for name, handler in handlers.items()
            }
        except Exception:
            connection.close()
            raise
        self._connection = connection
        self._handlers = handlers
        self._consumers = consumers
        return consumers


def _add_stats(
    total: CoreProjectionWorkerStats,
    current: ProjectionConsumerStats,
) -> CoreProjectionWorkerStats:
    return CoreProjectionWorkerStats(
        fetched=total.fetched + current.fetched,
        persisted=total.persisted + current.persisted,
        acknowledged=total.acknowledged + current.acknowledged,
        processed=total.processed + current.processed,
        failed=total.failed + current.failed,
    )


__all__ = ["CoreProjectionWorker", "CoreProjectionWorkerStats"]
