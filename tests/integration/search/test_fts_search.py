"""Integration tests for full-text search adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ledgermind_core.domain.events import KnowledgeCreated

from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.projections import KnowledgeFTSProjection
from ledgermind_local.search import SQLiteKnowledgeSearchAdapter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        }
    )
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id=memory_space_id,
        aggregate_id=knowledge_id,
        payload_json=payload,
    )


def _search_ids(search: SQLiteKnowledgeSearchAdapter, *, space: str, query: str, limit: int, offset: int = 0) -> list[str]:
    return [hit.knowledge_id for hit in search.search(space, query, limit=limit, offset=offset)]


def test_search_short_and_long_queries_share_fts_semantics(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-1",
        title="Search quality",
        target="engineering",
        statement="Canonical phrase keeps search stable.",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-1")

    adapter = SQLiteKnowledgeSearchAdapter(connection)
    short = adapter.search("space-a", "canonical", limit=10)
    long = adapter.search("space-a", 'canonical and and', limit=10)
    assert [item.knowledge_id for item in short] == [item.knowledge_id for item in long]


def test_search_respects_limit_and_offset(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")

    for index, statement in enumerate(
        (
            "shared anchor anchor anchor",
            "shared anchor",
            "shared anchor",
            "shared anchor",
            "shared anchor",
            "shared anchor",
        ),
        start=1,
    ):
        _add_knowledge(
            connection,
            memory_space_id="space-a",
            knowledge_id=f"k-{index}",
            title="Shared result",
            target="engineering",
            statement=statement,
        )

    projection = KnowledgeFTSProjection(connection)
    for index in range(1, 7):
        _emit_created(projection, memory_space_id="space-a", knowledge_id=f"k-{index}")

    search = SQLiteKnowledgeSearchAdapter(connection)

    page_1 = _search_ids(search, space="space-a", query="shared", limit=2, offset=0)
    page_2 = _search_ids(search, space="space-a", query="shared", limit=2, offset=2)
    all_first_4 = _search_ids(search, space="space-a", query="shared", limit=4, offset=0)

    assert page_1 == all_first_4[:2]
    assert page_2 == all_first_4[2:4]


def test_search_does_not_mutate_database(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-1",
        title="No mutation",
        target="engineering",
        statement="Read-only search should not write data.",
    )
    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-1")

    search = SQLiteKnowledgeSearchAdapter(connection)
    before_fts = connection.execute(
        "SELECT COUNT(*) AS total FROM knowledge_fts"
    ).fetchone()["total"]
    before_projection_state = connection.execute(
        "SELECT COUNT(*) AS total FROM projection_state"
    ).fetchone()["total"]
    before_changes = connection.total_changes

    _search_ids(search, space="space-a", query="search", limit=10, offset=0)
    _search_ids(search, space="space-a", query="read-only", limit=10, offset=0)

    after_fts = connection.execute(
        "SELECT COUNT(*) AS total FROM knowledge_fts"
    ).fetchone()["total"]
    after_projection_state = connection.execute(
        "SELECT COUNT(*) AS total FROM projection_state"
    ).fetchone()["total"]

    assert after_fts == before_fts
    assert after_projection_state == before_projection_state
    assert connection.total_changes == before_changes


def test_search_is_isolated_by_space_and_scoring(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")
    _ensure_space(connection, "space-b")

    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="Space A",
        target="engineering",
        statement="shared phrase for scoring",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a-2",
        title="Space A",
        target="engineering",
        statement="shared phrase in space a",
    )

    projection = KnowledgeFTSProjection(connection)
    for knowledge_id in ("k-a", "k-a-2"):
        _emit_created(projection, memory_space_id="space-a", knowledge_id=knowledge_id)

    search = SQLiteKnowledgeSearchAdapter(connection)

    baseline = search.search("space-a", "shared phrase", limit=3)
    baseline_scores = [item.lexical_score for item in baseline]

    for index in range(10):
        _add_knowledge(
            connection,
            memory_space_id="space-b",
            knowledge_id=f"k-b-{index}",
            title="Space B",
            target="engineering",
            statement="shared shared shared shared shared",
        )
        _emit_created(projection, memory_space_id="space-b", knowledge_id=f"k-b-{index}")

    updated = search.search("space-a", "shared phrase", limit=3)
    updated_scores = [item.lexical_score for item in updated]

    assert baseline_scores == updated_scores
    assert all(item.knowledge_id.startswith("k-a") for item in updated)


def test_search_filters_replaced_knowledge_after_stale_projection(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-old",
        title="Legacy fact",
        target="engineering",
        statement="legacy answer",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-new",
        title="Canonical answer",
        target="engineering",
        statement="legacy replacement",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-old")
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-new")

    connection.execute(
        """
        UPDATE knowledge_items
        SET superseded_by_id = ?, updated_at = ?
        WHERE memory_space_id = ? AND knowledge_id = ?
        """,
        ("k-new", _now(), "space-a", "k-old"),
    )
    # no projection event for supersede to keep projection stale and verify canonical filtering

    search = SQLiteKnowledgeSearchAdapter(connection)
    ids = _search_ids(search, space="space-a", query="legacy", limit=10)

    assert ids == ["k-new"]
