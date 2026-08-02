"""Integration tests for hybrid local knowledge search."""

from __future__ import annotations

from collections.abc import Sequence

from persistence import (
    migrations,
    open_sqlite_connection,
)
from projections import VectorProjectionStore
from projections.vectorizer import Vectorizer
from search import HybridKnowledgeSearchAdapter


def _build_connection(path):
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _build_vectorizer(mapping: dict[str, Sequence[float]]) -> Vectorizer:
    class _RecordingVectorizer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        @property
        def fingerprint(self) -> str:
            return "integration-hybrid-fake"

        @property
        def dimension(self) -> int:
            return 2

        def encode(self, texts: Sequence[str]) -> list[tuple[float, float]]:
            query = str(texts[0]) if texts else ""
            self.calls.append((query,))
            return [tuple(float(value) for value in mapping.get(query, (0.0, 0.0)))]

        def close(self) -> None:
            self.calls.append(("close",))

    return _RecordingVectorizer()


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
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )


def _add_knowledge(
    connection,
    *,
    knowledge_id: str,
    memory_space_id: str,
    title: str,
    target: str,
    statement: str,
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
            "",
            "pattern",
            1,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            None,
            None,
        ),
    )


def _build_fts_projection(connection, memory_space_id: str, knowledge_id: str) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO knowledge_fts (
            knowledge_id,
            memory_space_id,
            title,
            target,
            statement,
            rationale
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            knowledge_id,
            memory_space_id,
            connection.execute(
                """
                SELECT title FROM knowledge_items
                WHERE knowledge_id = ? AND memory_space_id = ?
                """,
                (knowledge_id, memory_space_id),
            ).fetchone()["title"],
            connection.execute(
                """
                SELECT target FROM knowledge_items
                WHERE knowledge_id = ? AND memory_space_id = ?
                """,
                (knowledge_id, memory_space_id),
            ).fetchone()["target"],
            connection.execute(
                """
                SELECT statement FROM knowledge_items
                WHERE knowledge_id = ? AND memory_space_id = ?
                """,
                (knowledge_id, memory_space_id),
            ).fetchone()["statement"],
            "",
        ),
    )


def test_hybrid_search_merges_fts_and_vector_results(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")
    _ensure_space(connection, "space-b")

    _add_knowledge(
        connection,
        knowledge_id="k-fts",
        memory_space_id="space-a",
        title="Canonical phrase",
        target="engineering",
        statement="shared anchor",
    )
    _add_knowledge(
        connection,
        knowledge_id="k-vector",
        memory_space_id="space-a",
        title="Vector only fact",
        target="engineering",
        statement="unrelated payload",
    )
    _add_knowledge(
        connection,
        knowledge_id="k-other",
        memory_space_id="space-b",
        title="Other space",
        target="engineering",
        statement="unrelated payload",
    )

    _build_fts_projection(connection, "space-a", "k-fts")

    vector_store_root = tmp_path / "vectors"
    store = VectorProjectionStore(vector_store_root)
    store.upsert(
        ("k-fts", "k-vector", "k-other"),
        ((-1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
    )
    store.close()

    vectorizer = _build_vectorizer({"shared": (1.0, 0.0)})
    adapter = HybridKnowledgeSearchAdapter(
        connection=connection,
        vector_store_root=vector_store_root,
        vectorizer_factory=lambda: vectorizer,
    )

    results = adapter.search("space-a", query="shared", limit=10)
    result_map = {item.knowledge_id: item for item in results}

    assert set(result_map) == {"k-fts", "k-vector"}
    assert result_map["k-vector"].lexical_score == 0.0
    assert result_map["k-vector"].vector_score == 1.0
    assert result_map["k-fts"].vector_score == 0.0
    assert result_map["k-fts"].lexical_score > 0.0


def test_hybrid_search_falls_back_to_fts_when_vectorizer_fails(tmp_path) -> None:
    connection = _build_connection(tmp_path / "state.db")
    _ensure_space(connection, "space-a")

    _add_knowledge(
        connection,
        knowledge_id="k-fts",
        memory_space_id="space-a",
        title="Canonical phrase",
        target="engineering",
        statement="shared anchor",
    )
    _build_fts_projection(connection, "space-a", "k-fts")

    vector_store_root = tmp_path / "vectors"
    store = VectorProjectionStore(vector_store_root)
    store.upsert(("k-vector",), ((1.0, 0.0),))
    store.close()

    def _broken_factory() -> Vectorizer:
        raise RuntimeError("llm model unavailable")

    adapter = HybridKnowledgeSearchAdapter(
        connection=connection,
        vector_store_root=vector_store_root,
        vectorizer_factory=_broken_factory,
    )
    results = adapter.search("space-a", query="shared", limit=10)

    assert len(results) == 1
    assert results[0].knowledge_id == "k-fts"
    assert results[0].vector_score is None
