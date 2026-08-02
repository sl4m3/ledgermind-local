"""Integration tests for durable projection outbox dispatcher and worker."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from projections.dispatcher import ProjectionDispatcher
from scheduler import OutboxWorker
from persistence import OutboxEvent, SQLiteUnitOfWork, migrations


def _build_time(value: int) -> str:
    return datetime(2026, 8, 1, tzinfo=timezone.utc, hour=0, minute=0, second=value).isoformat()


def _bootstrap_database(path) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.commit()


def _outbox_event(
    *,
    event_id: str = "evt-1",
    event_type: str = "knowledge.created",
    aggregate_id: str = "k-1",
    memory_space_id: str = "space-a",
    payload_json: str = '{"event_type":"knowledge.created","aggregate_id":"k-1"}',
    occurred_at: str = "2026-08-01T00:00:00+00:00",
    available_at: str = "2026-08-01T00:00:00+00:00",
    attempts: int = 0,
) -> OutboxEvent:
    return OutboxEvent(
        event_id=event_id,
        event_type=event_type,
        aggregate_id=aggregate_id,
        memory_space_id=memory_space_id,
        payload_json=payload_json,
        occurred_at=occurred_at,
        available_at=available_at,
        attempts=attempts,
        claimed_at=None,
        claimed_by=None,
        processed_at=None,
        last_error=None,
    )


def _seed_event(path, *, event_id: str, projection_names: tuple[str, ...]) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.outbox.add(
            _outbox_event(event_id=event_id),
            projection_names=projection_names,
        )
        uow.commit()


def _read_scalar(path, query: str, *, args: tuple = ()) -> str | None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(query, args).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


class _RecordingProjection:
    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str,
    ) -> bool:
        self.events.append(
            {
                "event_type": event_type,
                "memory_space_id": memory_space_id,
                "aggregate_id": aggregate_id,
                "payload_json": payload_json,
            }
        )
        return True


@dataclass
class _FlakyProjection:
    fail_times: int
    call_count: int = 0

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str,
    ) -> bool:
        self.call_count += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("projection failure")
        return True


def test_outbox_worker_delivers_to_single_projection(tmp_path) -> None:
    db = tmp_path / "state.db"
    projection = _RecordingProjection()
    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-1",
    )

    _seed_event(db, event_id="evt-single", projection_names=("projections.search",))
    assert worker.run_once() is True

    assert len(projection.events) == 1
    assert projection.events[0]["event_type"] == "knowledge.created"
    assert projection.events[0]["aggregate_id"] == "k-1"
    assert _read_scalar(
        db,
        "SELECT processed_at FROM outbox_events WHERE event_id = ?",
        args=("evt-single",),
    ) is not None


def test_outbox_worker_delivers_event_to_multiple_projections(tmp_path) -> None:
    db = tmp_path / "state.db"
    first = _RecordingProjection()
    second = _RecordingProjection()
    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher(
            {
                "projections.search": first,
                "projections.knowledge": second,
            }
        ),
        worker_id="worker-1",
    )

    _seed_event(
        db,
        event_id="evt-multi",
        projection_names=("projections.search", "projections.knowledge"),
    )
    assert worker.run_once() is True
    assert len(first.events) == 1
    assert len(second.events) == 1
    assert _read_scalar(
        db,
        "SELECT processed_at FROM outbox_events WHERE event_id = ?",
        args=("evt-multi",),
    ) is not None


def test_outbox_worker_failure_of_one_projection_does_not_block_other(tmp_path) -> None:
    db = tmp_path / "state.db"
    failing = _FlakyProjection(fail_times=1)
    passing = _RecordingProjection()
    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher(
            {
                "projections.search": failing,
                "projections.knowledge": passing,
            }
        ),
        worker_id="worker-1",
    )

    _seed_event(
        db,
        event_id="evt-failure",
        projection_names=("projections.search", "projections.knowledge"),
    )
    assert worker.run_once() is True

    assert failing.call_count == 1
    assert len(passing.events) == 1
    assert _read_scalar(
        db,
        "SELECT processed_at FROM projection_deliveries WHERE projection_name = ? AND event_id = ?",
        args=("projections.search", "evt-failure"),
    ) is None
    assert _read_scalar(
        db,
        "SELECT processed_at FROM projection_deliveries WHERE projection_name = ? AND event_id = ?",
        args=("projections.knowledge", "evt-failure"),
    ) is not None
    assert _read_scalar(
        db,
        "SELECT processed_at FROM outbox_events WHERE event_id = ?",
        args=("evt-failure",),
    ) is None


def test_outbox_worker_retries_failed_delivery_and_replays(tmp_path) -> None:
    db = tmp_path / "state.db"
    projection = _FlakyProjection(fail_times=1)

    now = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    _seed_event(
        db,
        event_id="evt-retry",
        projection_names=("projections.search",),
    )

    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-fail",
        now_factory=lambda: now,
    )
    assert worker.run_once() is True
    assert projection.call_count == 1
    assert _read_scalar(
        db,
        "SELECT processed_at FROM projection_deliveries WHERE projection_name = ? AND event_id = ?",
        args=("projections.search", "evt-retry"),
    ) is None

    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-success",
        now_factory=lambda: now + timedelta(seconds=2),
    )
    assert worker.run_once() is True
    assert projection.call_count == 2
    assert _read_scalar(
        db,
        "SELECT processed_at FROM projection_deliveries WHERE projection_name = ? AND event_id = ?",
        args=("projections.search", "evt-retry"),
    ) is not None


def test_outbox_worker_releases_stale_claim(tmp_path) -> None:
    db = tmp_path / "state.db"
    projection = _RecordingProjection()
    now = datetime(2026, 8, 1, 0, 1, 0, tzinfo=timezone.utc)

    with SQLiteUnitOfWork(db) as uow:
        migrations.apply_migrations(uow.connection)
        uow.outbox.add(
            _outbox_event(event_id="evt-stale"),
            projection_names=("projections.search",),
        )
        uow.connection.execute(
            """
            UPDATE projection_deliveries
            SET claimed_at = ?, claimed_by = ?
            WHERE projection_name = ? AND event_id = ?
            """,
            (_build_time(10), "dead-worker", "projections.search", "evt-stale"),
        )
        uow.commit()

    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-recover",
        now_factory=lambda: now,
        stale_claim_ttl_seconds=30,
    )
    assert worker.run_once() is True

    assert len(projection.events) == 1
    assert _read_scalar(
        db,
        "SELECT processed_at FROM projection_deliveries WHERE projection_name = ? AND event_id = ?",
        args=("projections.search", "evt-stale"),
    ) is not None


def test_outbox_worker_stops_gracefully(tmp_path) -> None:
    db = tmp_path / "state.db"
    _bootstrap_database(db)
    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": _RecordingProjection()}),
        worker_id="worker-stop",
        poll_interval_seconds=1.0,
    )

    thread = threading.Thread(target=worker.run)
    thread.start()
    worker.request_stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()


def test_outbox_event_survives_worker_restart(tmp_path) -> None:
    db = tmp_path / "state.db"
    projection = _FlakyProjection(fail_times=1)

    _seed_event(
        db,
        event_id="evt-restart",
        projection_names=("projections.search",),
    )

    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-1",
        now_factory=lambda: datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    assert worker.run_once() is True
    assert projection.call_count == 1
    assert _read_scalar(
        db,
        "SELECT processed_at FROM outbox_events WHERE event_id = ?",
        args=("evt-restart",),
    ) is None

    worker = OutboxWorker(
        database_path=db,
        dispatcher=ProjectionDispatcher({"projections.search": projection}),
        worker_id="worker-2",
        now_factory=lambda: datetime(2026, 8, 1, 0, 0, 5, tzinfo=timezone.utc),
    )
    assert worker.run_once() is True
    assert projection.call_count == 2
    assert _read_scalar(
        db,
        "SELECT processed_at FROM outbox_events WHERE event_id = ?",
        args=("evt-restart",),
    ) is not None
