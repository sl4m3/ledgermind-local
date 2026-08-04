from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
)
from ledgermind_local.persistence import rounds_migrations
from ledgermind_local.projections.vector import KnowledgeVectorProjection


class FakeVectorizer:
    dimension = 2
    fingerprint = "fake-fingerprint"
    model_name = "fake-model"

    def encode(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        return None


def test_vector_core_event_does_not_load_local_knowledge() -> None:
    connection = sqlite3.connect(":memory:")
    rounds_migrations.apply_migrations(connection)
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=Path("/tmp/ledgermind-core-event-vector-test"),
        vectorizer_factory=FakeVectorizer,
    )
    event = CoreProjectionEvent(
        event_id="event-vector-1",
        memory_space_id="space-1",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-1",
                "title": "Vector title",
                "target": "projection",
                "statement": "Vector event",
                "projection_version": 1,
            },
            separators=(",", ":"),
        ),
        occurred_at="2026-01-01T00:00:00Z",
    )

    assert projection.handle_core_event(event) is True
    projection.close()
    connection.close()
