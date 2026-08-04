from __future__ import annotations

import json
import threading
from pathlib import Path

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    PollProjectionEventsResult,
)
from ledgermind_local.persistence import open_sqlite_connection, rounds_migrations
from ledgermind_local.scheduler.core_projection_worker import CoreProjectionWorker


class FakeGateway:
    def __init__(self, event: CoreProjectionEvent) -> None:
        self.event = event
        self.polls = []
        self.acks = []
        self.used: set[str] = set()

    def poll_projection_events(self, command):
        self.polls.append(command)
        if command.consumer_id in self.used:
            return PollProjectionEventsResult(events=(), has_more=False)
        self.used.add(command.consumer_id)
        return PollProjectionEventsResult(events=(self.event,), has_more=False)

    def ack_projection_events(self, command):
        self.acks.append(command)
        return type("Ack", (), {"acknowledged": command.event_ids})()


class Handler:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def handle_core_event(self, event: CoreProjectionEvent) -> bool:
        self.calls.append(event.event_id)
        return True


def _event() -> CoreProjectionEvent:
    return CoreProjectionEvent(
        event_id="event-1",
        memory_space_id="space-1",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-1",
                "title": "Title",
                "target": "Target",
                "statement": "Statement",
                "projection_version": 1,
            },
            separators=(",", ":"),
        ),
        occurred_at="2026-01-01T00:00:00Z",
    )


def test_core_projection_worker_polls_local_memory_spaces_and_applies_events(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "rounds.db"
    with open_sqlite_connection(database_path) as connection:
        rounds_migrations.apply_migrations(connection)
        connection.execute(
            """
            INSERT INTO memory_spaces (
                memory_space_id, display_name, source_client, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("space-1", "Space", "test", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        connection.commit()

    gateway = FakeGateway(_event())
    calls: list[str] = []
    worker = CoreProjectionWorker(
        database_path=database_path,
        gateway=gateway,
        consumer_id="local-projections",
        handlers_factory=lambda _connection: {"fts": Handler(calls)},
    )

    stats = worker.process_once()

    assert stats.fetched == 1
    assert stats.acknowledged == 1
    assert stats.processed == 1
    assert calls == ["event-1"]
    assert [command.memory_space_id for command in gateway.polls] == ["space-1"]
    assert [command.consumer_id for command in gateway.polls] == ["local-projections:fts"]
    worker.close()


def test_core_projection_worker_keeps_projection_consumers_independent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "rounds.db"
    with open_sqlite_connection(database_path) as connection:
        rounds_migrations.apply_migrations(connection)
        connection.execute(
            """
            INSERT INTO memory_spaces (
                memory_space_id, display_name, source_client, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("space-1", "Space", "test", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        connection.commit()

    gateway = FakeGateway(_event())
    calls: list[str] = []
    worker = CoreProjectionWorker(
        database_path=database_path,
        gateway=gateway,
        consumer_id="local-projections",
        handlers_factory=lambda _connection: {
            "fts": Handler(calls),
            "vector": Handler(calls),
        },
    )

    stats = worker.process_once()

    assert stats.fetched == 2
    assert stats.acknowledged == 2
    assert stats.processed == 2
    assert calls == ["event-1", "event-1"]
    assert [command.consumer_id for command in gateway.polls] == [
        "local-projections:fts",
        "local-projections:vector",
    ]
    worker.close()


def test_core_projection_worker_loop_survives_unexpected_gateway_error(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "rounds.db"
    with open_sqlite_connection(database_path) as connection:
        rounds_migrations.apply_migrations(connection)
        connection.execute(
            """
            INSERT INTO memory_spaces (
                memory_space_id, display_name, source_client, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("space-1", "Space", "test", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        connection.commit()

    recovered = threading.Event()

    class FlakyGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__(_event())
            self.failed = False

        def poll_projection_events(self, command):
            if not self.failed:
                self.failed = True
                raise RuntimeError("event payload must not be logged")
            recovered.set()
            return super().poll_projection_events(command)

    gateway = FlakyGateway()
    worker = CoreProjectionWorker(
        database_path=database_path,
        gateway=gateway,
        consumer_id="local-projections",
        handlers_factory=lambda _connection: {"fts": Handler([])},
    )
    loop = worker.create_loop(
        poll_interval_seconds=0,
        initial_backoff_seconds=0,
        max_backoff_seconds=0,
    )

    loop.start()
    assert recovered.wait(timeout=1)
    loop.request_stop()
    assert loop.join(timeout=1) is True
    assert worker.state.failed_count == 1
    assert worker.state.healthy is True
    worker.close()
