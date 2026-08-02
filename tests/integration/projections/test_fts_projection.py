"""Integration tests for SQLite FTS knowledge projection."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from domain.events import KnowledgeCreated, KnowledgeDeleted, KnowledgeSuperseded

from persistence import migrations, open_sqlite_connection
from projections.fts import KnowledgeFTSProjection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bootstrap(path) -> None:
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _build_connection(path):
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _ensure_space(connection, memory_space_id: str) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO memory_spaces (
            memory_space_id,
            display_name,
            source_client,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            memory_space_id,
            None,
            "hermes",
            _now(),
            _now(),
        ),
    )


def _add_knowledge(
    connection,
    *,
    memory_space_id: str,
    knowledge_id: str,
    title: str,
    target: str,
    statement: str,
    rationale: str = "",
    superseded_by_id: str | None = None,
    deleted_at: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO knowledge_items (
            knowledge_id,
            memory_space_id,
            title,
            target,
            statement,
            rationale,
            phase,
            version,
            created_at,
            updated_at,
            superseded_by_id,
            deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            knowledge_id,
            memory_space_id,
            title,
            target,
            statement,
            rationale,
            "pattern",
            1,
            _now(),
            _now(),
            superseded_by_id,
            deleted_at,
        ),
    )


def _update_statement(connection, *, memory_space_id: str, knowledge_id: str, statement: str) -> None:
    current = connection.execute(
        """
        SELECT version
        FROM knowledge_items
        WHERE memory_space_id = ? AND knowledge_id = ?
        """,
        (memory_space_id, knowledge_id),
    ).fetchone()
    if current is None:
        return

    connection.execute(
        """
        UPDATE knowledge_items
        SET statement = ?, updated_at = ?, version = version + 1
        WHERE memory_space_id = ? AND knowledge_id = ?
        """,
        (statement, _now(), memory_space_id, knowledge_id),
    )


def _set_superseded(
    connection,
    *,
    memory_space_id: str,
    knowledge_id: str,
    superseded_by_id: str,
) -> None:
    connection.execute(
        """
        UPDATE knowledge_items
        SET superseded_by_id = ?, updated_at = ?, version = version + 1
        WHERE memory_space_id = ? AND knowledge_id = ?
        """,
        (superseded_by_id, _now(), memory_space_id, knowledge_id),
    )


def _set_deleted(connection, *, memory_space_id: str, knowledge_id: str) -> None:
    connection.execute(
        """
        UPDATE knowledge_items
        SET deleted_at = ?, updated_at = ?
        WHERE memory_space_id = ? AND knowledge_id = ?
        """,
        (_now(), _now(), memory_space_id, knowledge_id),
    )


def _emit_created(
    projection: KnowledgeFTSProjection,
    *,
    memory_space_id: str,
    knowledge_id: str,
) -> None:
    payload = json.dumps(
        {
            "event_type": KnowledgeCreated.EVENT_NAME,
            "aggregate_id": knowledge_id,
        },
    )
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id=memory_space_id,
        aggregate_id=knowledge_id,
        payload_json=payload,
    )


def _emit_superseded(
    projection: KnowledgeFTSProjection,
    *,
    memory_space_id: str,
    old_knowledge_id: str,
    new_knowledge_id: str,
) -> None:
    payload = json.dumps(
        {
            "event_type": KnowledgeSuperseded.EVENT_NAME,
            "previous_knowledge_id": old_knowledge_id,
            "next_knowledge_id": new_knowledge_id,
        },
    )
    projection.handle_event(
        event_type=KnowledgeSuperseded.EVENT_NAME,
        memory_space_id=memory_space_id,
        aggregate_id=old_knowledge_id,
        payload_json=payload,
    )


def _emit_deleted(
    projection: KnowledgeFTSProjection,
    *,
    memory_space_id: str,
    knowledge_id: str,
) -> None:
    payload = json.dumps(
        {
            "event_type": KnowledgeDeleted.EVENT_NAME,
            "knowledge_id": knowledge_id,
            "by_atom_id": "atom-1",
        },
    )
    projection.handle_event(
        event_type=KnowledgeDeleted.EVENT_NAME,
        memory_space_id=memory_space_id,
        aggregate_id=knowledge_id,
        payload_json=payload,
    )


def _search_hit_ids(connection, projection: KnowledgeFTSProjection, *, space: str, query: str) -> list[str]:
    return [item.knowledge_id for item in projection.search(space, query, limit=10)]


def _count_fts(connection, *, memory_space_id: str | None = None) -> int:
    if memory_space_id is None:
        return int(
            connection.execute("SELECT COUNT(*) AS total FROM knowledge_fts").fetchone()["total"]
        )
    return int(
        connection.execute(
            "SELECT COUNT(*) AS total FROM knowledge_fts WHERE memory_space_id = ?",
            (memory_space_id,),
        ).fetchone()["total"]
    )


def test_projection_creates_knowledge_in_search_index(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-1",
        title="Pattern storage",
        target="architecture",
        statement="How to keep canonical facts",
    )
    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-1")

    assert _search_hit_ids(connection, projection, space="space-a", query="canonical") == ["k-1"]


def test_projection_updates_existing_knowledge_by_recreate_event(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-2",
        title="Pattern storage",
        target="architecture",
        statement="old statement",
    )
    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-2")

    assert _search_hit_ids(connection, projection, space="space-a", query="old") == ["k-2"]

    _update_statement(connection, memory_space_id="space-a", knowledge_id="k-2", statement="updated statement")
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-2")

    assert _search_hit_ids(connection, projection, space="space-a", query="old") == []
    assert _search_hit_ids(connection, projection, space="space-a", query="updated") == ["k-2"]


def test_projection_supersede_removes_old_knowledge_from_current_search(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-old",
        title="Old pattern",
        target="architecture",
        statement="legacy answer",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-new",
        title="New pattern",
        target="architecture",
        statement="replacement answer",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-old")
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-new")

    assert _search_hit_ids(connection, projection, space="space-a", query="legacy") == ["k-old"]

    _set_superseded(connection, memory_space_id="space-a", knowledge_id="k-old", superseded_by_id="k-new")
    _emit_superseded(
        projection,
        memory_space_id="space-a",
        old_knowledge_id="k-old",
        new_knowledge_id="k-new",
    )

    assert _search_hit_ids(connection, projection, space="space-a", query="legacy") == []
    assert _search_hit_ids(connection, projection, space="space-a", query="replacement") == ["k-new"]


def test_projection_delete_removes_knowledge_from_search(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-3",
        title="Pattern storage",
        target="architecture",
        statement="to be deleted",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-3")
    assert _search_hit_ids(connection, projection, space="space-a", query="deleted") == ["k-3"]

    _set_deleted(connection, memory_space_id="space-a", knowledge_id="k-3")
    _emit_deleted(projection, memory_space_id="space-a", knowledge_id="k-3")
    assert _search_hit_ids(connection, projection, space="space-a", query="deleted") == []


def test_projection_duplicate_events_are_idempotent(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-dup",
        title="Pattern storage",
        target="architecture",
        statement="duplicate",
    )
    projection = KnowledgeFTSProjection(connection)

    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-dup")
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-dup")

    assert _count_fts(connection, memory_space_id="space-a") == 1
    assert _search_hit_ids(connection, projection, space="space-a", query="duplicate") == ["k-dup"]


def test_projection_rebuilds_full_index(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _ensure_space(connection, "space-b")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="Pattern storage",
        target="architecture",
        statement="current-a",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a-old",
        title="Pattern storage",
        target="architecture",
        statement="old-a",
        superseded_by_id="k-a",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-b",
        knowledge_id="k-b",
        title="Pattern storage",
        target="architecture",
        statement="current-b",
        deleted_at=_now(),
    )

    connection.execute(
        """
        INSERT INTO knowledge_fts (
            knowledge_id,
            memory_space_id,
            title,
            target,
            statement,
            rationale
        ) VALUES ('stale', 'space-a', 'stale', 'stale', 'stale', 'stale')
        """
    )
    projection = KnowledgeFTSProjection(connection)
    count = projection.rebuild()

    assert count == 1
    assert _search_hit_ids(connection, projection, space="space-a", query="current-a") == ["k-a"]
    assert _search_hit_ids(connection, projection, space="space-a", query="old-a") == []
    state = connection.execute(
        "SELECT item_count FROM projection_state WHERE projection_name = ?",
        (projection.projection_name,),
    ).fetchone()
    assert state is not None
    assert state["item_count"] == 1


def test_projection_is_isolated_by_memory_space(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _ensure_space(connection, "space-b")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="Pattern storage",
        target="architecture",
        statement="shared token",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-b",
        knowledge_id="k-b",
        title="Pattern storage",
        target="architecture",
        statement="shared token",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-a")
    _emit_created(projection, memory_space_id="space-b", knowledge_id="k-b")

    assert _search_hit_ids(connection, projection, space="space-a", query="shared") == ["k-a"]
    assert _search_hit_ids(connection, projection, space="space-b", query="shared") == ["k-b"]


def test_projection_supports_unicode_russian_text(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-rus",
        title="Память и контур",
        target="архитектура",
        statement="Поиск по русскому тексту должен работать.",
    )
    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-rus")

    assert _search_hit_ids(connection, projection, space="space-a", query="русскому") == ["k-rus"]


def test_projection_special_query_syntax_does_not_raise_errors(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-safe",
        title="Pattern storage",
        target="architecture",
        statement='canonical deletion "quoted"',
    )
    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-safe")

    assert _search_hit_ids(connection, projection, space="space-a", query='canonical" OR deletion') == ["k-safe"]
