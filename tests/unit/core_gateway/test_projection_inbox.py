from __future__ import annotations

import json

import pytest

from ledgermind_local.core_gateway.projection_contracts import CoreProjectionEvent
from ledgermind_local.core_gateway.projection_inbox import CoreProjectionInbox
from ledgermind_local.persistence import open_sqlite_connection, rounds_migrations


def _event(event_id: str = "event-1") -> CoreProjectionEvent:
    return CoreProjectionEvent(
        event_id=event_id,
        memory_space_id="space-a",
        aggregate_id="knowledge-1",
        event_type="knowledge_projection_upsert",
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-a",
                "title": "Title",
                "target": "target",
                "statement": "Statement",
                "projection_version": 1,
            }
        ),
        occurred_at="2026-08-03T00:00:00+00:00",
    )


def _connection(path):
    connection = open_sqlite_connection(path)
    rounds_migrations.apply_migrations(connection)
    connection.commit()
    return connection


def test_inbox_persists_event_before_independent_projection_delivery(tmp_path) -> None:
    connection = _connection(tmp_path / "rounds.db")
    try:
        inbox = CoreProjectionInbox(connection)
        inbox.save_events([_event()], projection_names=("fts", "vector"))
        connection.commit()

        assert [item.event_id for item in inbox.ready("fts")] == ["event-1"]
        assert [item.event_id for item in inbox.ready("vector")] == ["event-1"]

        inbox.mark_processed("fts", "event-1")
        connection.commit()

        assert inbox.ready("fts") == []
        assert [item.event_id for item in inbox.ready("vector")] == ["event-1"]
    finally:
        connection.close()


def test_inbox_replay_is_idempotent_but_payload_conflict_is_rejected(tmp_path) -> None:
    connection = _connection(tmp_path / "rounds.db")
    try:
        inbox = CoreProjectionInbox(connection)
        inbox.save_events([_event()], projection_names=("fts",))
        inbox.save_events([_event()], projection_names=("fts",))
        connection.commit()

        conflicting = _event()
        conflicting = CoreProjectionEvent(
            event_id=conflicting.event_id,
            memory_space_id=conflicting.memory_space_id,
            aggregate_id=conflicting.aggregate_id,
            event_type=conflicting.event_type,
            payload_json=conflicting.payload_json.replace("Statement", "Other"),
            occurred_at=conflicting.occurred_at,
        )
        with pytest.raises(ValueError, match="conflicts with stored event"):
            inbox.save_events([conflicting], projection_names=("fts",))
    finally:
        connection.close()
