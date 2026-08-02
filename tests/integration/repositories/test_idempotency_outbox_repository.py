"""Integration tests for SQLite idempotency and outbox repositories."""

from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.persistence import (
    OutboxEvent,
    SQLiteUnitOfWork,
    StoredIdempotencyResult,
    migrations,
)


def _idempotency_payload(
    *,
    memory_space_id: str = "space-a",
    key: str = "idem-1",
    request_hash: str = "sha256:" + "a" * 64,
    response_json: str = '{"projections_pending":true}',
    created_at: str = "2026-08-01T00:00:00Z",
    expires_at: str | None = None,
) -> StoredIdempotencyResult:
    return StoredIdempotencyResult(
        memory_space_id=memory_space_id,
        key=key,
        request_hash=request_hash,
        response_json=response_json,
        created_at=created_at,
        expires_at=expires_at,
    )


def _outbox_event(
    *,
    event_id: str = "evt-1",
    event_type: str = "atom.created",
    aggregate_id: str = "atom-1",
    memory_space_id: str = "space-a",
    payload_json: str = '{"event_type":"atom.created","aggregate_id":"atom-1"}',
    occurred_at: str = "2026-08-01T00:00:00Z",
    available_at: str = "2026-08-01T00:00:00Z",
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


def test_idempotency_store_and_read(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.idempotency.add(_idempotency_payload())
        assert uow.idempotency.get("space-a", "idem-1") == _idempotency_payload()
        uow.commit()


def test_idempotency_conflict_is_raised(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.idempotency.add(_idempotency_payload())
        with pytest.raises(sqlite3.IntegrityError):
            uow.idempotency.add(_idempotency_payload())
        uow.rollback()


def test_outbox_event_and_deliveries_persist_in_same_transaction(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        event = _outbox_event()
        uow.outbox.add(event, projection_names=("p-atoms", "p-knowledge"))
        uow.commit()

    conn = sqlite3.connect(db_path)
    try:
        events = conn.execute(
            "SELECT COUNT(*) AS total FROM outbox_events"
        ).fetchone()[0]
        deliveries = conn.execute(
            "SELECT COUNT(*) AS total FROM projection_deliveries"
        ).fetchone()[0]
        assert events == 1
        assert deliveries == 2
    finally:
        conn.close()


def test_outbox_rollback_removes_event_and_deliveries(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as setup:
        migrations.apply_migrations(setup.connection)
        setup.commit()

    with pytest.raises(sqlite3.IntegrityError), SQLiteUnitOfWork(db_path) as uow:
        event = _outbox_event(event_id="evt-1")
        uow.outbox.add(event, projection_names=("projection",))
        uow.outbox.add(event, projection_names=("projection",))

    conn = sqlite3.connect(db_path)
    try:
        events = conn.execute(
            "SELECT COUNT(*) AS total FROM outbox_events"
        ).fetchone()[0]
        deliveries = conn.execute(
            "SELECT COUNT(*) AS total FROM projection_deliveries"
        ).fetchone()[0]
        assert events == 0
        assert deliveries == 0
    finally:
        conn.close()


def test_outbox_ready_selection_respects_available_at(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.outbox.add(
            _outbox_event(event_id="e1", available_at="2026-08-01T00:00:00Z"),
            projection_names=("projection",),
        )
        uow.outbox.add(
            _outbox_event(event_id="e2", available_at="2026-08-01T00:10:00Z"),
            projection_names=("projection",),
        )
        uow.commit()

    with SQLiteUnitOfWork(db_path) as uow:
        ready = uow.outbox.list_ready("projection", now="2026-08-01T00:05:00Z")
        assert len(ready) == 1
        assert ready[0].event_id == "e1"


def test_outbox_stale_claim_is_released(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.outbox.add(
            _outbox_event(
                event_id="e1",
                available_at="2026-08-01T00:00:00Z",
                occurred_at="2026-08-01T00:00:00Z",
            ),
            projection_names=("projection",),
        )
        uow.commit()

    with SQLiteUnitOfWork(db_path) as uow:
        uow.connection.execute(
            """
            UPDATE projection_deliveries
            SET claimed_at = ?, claimed_by = ?
            WHERE projection_name = ? AND event_id = ?
            """,
            ("2026-08-01T00:00:10Z", "old-worker", "projection", "e1"),
        )

        acquired = uow.outbox.acquire_next(
            "projection",
            worker_id="new-worker",
            now="2026-08-01T00:01:00Z",
            stale_claim_before="2026-08-01T00:00:30Z",
        )
        assert acquired is not None
        assert acquired.event_id == "e1"
        assert acquired.claimed_by == "new-worker"
