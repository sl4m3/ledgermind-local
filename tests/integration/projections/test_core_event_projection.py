from __future__ import annotations

import json
import sqlite3

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
)
from ledgermind_local.persistence import rounds_migrations
from ledgermind_local.projections.fts import KnowledgeFTSProjection


def test_fts_core_event_projects_without_reading_local_knowledge_items() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    rounds_migrations.apply_migrations(connection)
    projection = KnowledgeFTSProjection(connection)

    def forbidden_loader(*, knowledge_id: str, memory_space_id: str):
        raise AssertionError("Core event projection must not read knowledge_items")

    projection._load_knowledge = forbidden_loader  # type: ignore[method-assign]
    event = CoreProjectionEvent(
        event_id="event-fts-1",
        memory_space_id="space-1",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-1",
                "title": "Local FTS title",
                "target": "projection",
                "statement": "Core event is the source",
                "projection_version": 1,
            },
            separators=(",", ":"),
        ),
        occurred_at="2026-01-01T00:00:00Z",
    )

    assert projection.handle_core_event(event) is True
    hits = projection.search("space-1", "Core event source", limit=5)
    assert [hit.knowledge_id for hit in hits] == ["knowledge-1"]
    connection.close()
