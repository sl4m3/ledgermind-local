from __future__ import annotations

import json
import sqlite3

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
)
from ledgermind_local.persistence import rounds_migrations
from ledgermind_local.projections.markdown import KnowledgeMarkdownProjection


def test_markdown_core_event_uses_only_public_projection_payload(tmp_path) -> None:
    connection = sqlite3.connect(":memory:")
    rounds_migrations.apply_migrations(connection)
    projection = KnowledgeMarkdownProjection(
        connection=connection,
        markdown_root=tmp_path,
    )
    event = CoreProjectionEvent(
        event_id="event-markdown-1",
        memory_space_id="space-1",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-1",
                "title": "Markdown title",
                "target": "projection",
                "statement": "Markdown event",
                "projection_version": 1,
            },
            separators=(",", ":"),
        ),
        occurred_at="2026-01-01T00:00:00Z",
    )

    assert projection.handle_core_event(event) is True
    path = projection._entry_path("space-1", "knowledge-1")
    content = path.read_text(encoding="utf-8")
    assert "Markdown title" in content
    assert "Markdown event" in content
    assert "phase" not in content
    assert "rationale" not in content
    connection.close()
