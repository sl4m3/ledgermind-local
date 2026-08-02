"""Integration tests for local vector knowledge projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from domain.events import KnowledgeCreated, KnowledgeDeleted, KnowledgeSuperseded
from persistence import (
    Knowledge,
    SQLiteKnowledgeRepository,
    migrations,
    open_sqlite_connection,
)
from projections import KnowledgeVectorProjection, VectorProjectionStore


@dataclass
class _SpyVectorizerConfig:
    dimension: int = 2
    fingerprint: str = "unit-test-vectorizer"
    model_name: str | None = None


class _SpyVectorizer:
    def __init__(self, config: _SpyVectorizerConfig) -> None:
        self._config = config
        self.calls: list[tuple[str, ...]] = []
        self.closed = 0

    @property
    def fingerprint(self) -> str:
        return self._config.fingerprint

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model_name(self) -> str | None:
        return self._config.model_name

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        batch = tuple(texts)
        self.calls.append(batch)
        result: list[tuple[float, ...]] = []
        for text in texts:
            vector = [float(len(text)), float(len(texts))]
            if self._config.dimension < 2:
                vector = vector[: self._config.dimension]
            elif self._config.dimension > 2:
                vector = vector + [0.0] * (self._config.dimension - 2)
            result.append(tuple(vector[: self._config.dimension]))
        return [tuple(item) for item in result]

    def close(self) -> None:
        self.closed += 1


class _FailingSpyVectorizer(_SpyVectorizer):
    def __init__(self, config: _SpyVectorizerConfig, fail_on_call: int) -> None:
        super().__init__(config)
        self.fail_on_call = fail_on_call

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        batch = tuple(texts)
        self.calls.append(batch)
        if len(self.calls) >= self.fail_on_call:
            raise RuntimeError("simulated rebuild interruption")
        return super().encode(list(batch))


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
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
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
    SQLiteKnowledgeRepository(connection).add(
        Knowledge(
            knowledge_id=knowledge_id,
            memory_space_id=memory_space_id,
            title=title,
            target=target,
            statement=statement,
            rationale=rationale,
            phase="pattern",
            version=1,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            superseded_by_id=superseded_by_id,
            deleted_at=deleted_at,
        )
    )


def _emit_projection_event(
    projection: KnowledgeVectorProjection,
    *,
    event_type: str,
    memory_space_id: str,
    payload: dict[str, object] | None = None,
    aggregate_id: str = "k-default",
) -> bool:
    payload = dict(payload or {"event_type": event_type, "aggregate_id": aggregate_id})
    payload["event_type"] = event_type
    return projection.handle_event(
        event_type=event_type,
        memory_space_id=memory_space_id,
        aggregate_id=aggregate_id,
        payload_json=json.dumps(payload),
    )


def _read_root(path: Path) -> VectorProjectionStore:
    return VectorProjectionStore(path)


def test_projection_handles_created_updates_current_knowledge(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-1",
        title="Title",
        target="Target",
        statement="Statement",
        rationale="Rationale",
    )

    vector_store_root = tmp_path / "vectors"
    config = _SpyVectorizerConfig()
    vectorizer = _SpyVectorizer(config)
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=vector_store_root,
        vectorizer_factory=lambda: vectorizer,
    )

    changed = _emit_projection_event(
        projection,
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-1",
    )
    assert changed is True
    assert vectorizer.calls == [("Title Target Statement Rationale",)]

    store = _read_root(vector_store_root)
    assert store.ids == ("k-1",)
    assert store.vectors == ((32.0, 1.0),)


def test_projection_processes_supersede_and_delete_events(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-old",
        title="Old",
        target="T",
        statement="old version",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-new",
        title="New",
        target="T",
        statement="new version",
    )

    config = _SpyVectorizerConfig()
    vectorizer = _SpyVectorizer(config)
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: vectorizer,
    )

    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-old",
        payload_json=json.dumps({"event_type": KnowledgeCreated.EVENT_NAME, "aggregate_id": "k-old"}),
    )
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-new",
        payload_json=json.dumps({"event_type": KnowledgeCreated.EVENT_NAME, "aggregate_id": "k-new"}),
    )

    projection.handle_event(
        event_type=KnowledgeSuperseded.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-old",
        payload_json=json.dumps(
            {
                "event_type": KnowledgeSuperseded.EVENT_NAME,
                "previous_knowledge_id": "k-old",
                "next_knowledge_id": "k-new",
            }
        ),
    )

    store = _read_root(tmp_path / "vectors")
    assert store.ids == ("k-new",)

    _emit_projection_event(
        projection,
        event_type=KnowledgeDeleted.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-new",
        payload={"event_type": KnowledgeDeleted.EVENT_NAME, "knowledge_id": "k-new"},
    )

    store = _read_root(tmp_path / "vectors")
    assert store.ids == ()
    assert vectorizer.calls[0] == ("Old T old version",)
    assert vectorizer.calls[1] == ("New T new version",)
    assert vectorizer.calls[2] == ("New T new version",)


def test_projection_processes_updated_event(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-1",
        title="Title",
        target="Target",
        statement="First statement",
        rationale="Rationale",
    )

    vectorizer = _SpyVectorizer(_SpyVectorizerConfig())
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: vectorizer,
    )

    changed = _emit_projection_event(
        projection,
        event_type="knowledge.updated",
        memory_space_id="space-a",
        aggregate_id="k-1",
        payload={
            "event_type": "knowledge.updated",
            "knowledge_id": "k-1",
        },
    )

    store = _read_root(tmp_path / "vectors")
    assert changed is True
    assert store.ids == ("k-1",)
    assert vectorizer.calls == [("Title Target First statement Rationale",)]
    assert store.vectors == ((38.0, 1.0),)


def test_projection_is_idempotent_and_batch_capable(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="Alpha",
        target="A",
        statement="one",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-b",
        title="Beta",
        target="B",
        statement="two",
    )

    vectorizer = _SpyVectorizer(_SpyVectorizerConfig())
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: vectorizer,
    )

    changed_first = _emit_projection_event(
        projection,
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-a",
        payload={"event_type": KnowledgeCreated.EVENT_NAME, "knowledge_ids": ["k-a", "k-b"]},
    )
    changed_second = _emit_projection_event(
        projection,
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-a",
        payload={"event_type": KnowledgeCreated.EVENT_NAME, "knowledge_ids": ["k-a", "k-b"]},
    )

    assert changed_first is True
    assert changed_second is True

    store = _read_root(tmp_path / "vectors")
    assert store.ids == ("k-a", "k-b")
    assert len(vectorizer.calls) == 2
    assert vectorizer.calls[0] == ("Alpha A one", "Beta B two")
    assert vectorizer.calls[1] == ("Alpha A one", "Beta B two")


def test_projection_does_not_load_vectorizer_on_non_vectorizing_events(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")

    creation_counter: dict[str, int] = {"count": 0}

    def _factory() -> _SpyVectorizer:
        creation_counter["count"] += 1
        return _SpyVectorizer(_SpyVectorizerConfig())

    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=_factory,
    )

    changed = _emit_projection_event(
        projection,
        event_type=KnowledgeDeleted.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-missing",
        payload={"event_type": KnowledgeDeleted.EVENT_NAME, "knowledge_id": "k-missing"},
    )
    assert changed is False
    assert creation_counter["count"] == 0


def test_projection_rebuild_empty_database(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    vector_store_root = tmp_path / "vectors"
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=vector_store_root,
        vectorizer_factory=lambda: _SpyVectorizer(_SpyVectorizerConfig()),
    )

    count = projection.rebuild()
    projection.close()

    assert count == 0
    store = _read_root(vector_store_root)
    assert store.ids == ()
    assert store.document_count == 0
    assert store.manifest["document_count"] == 0
    assert store.manifest["model_fingerprint"] == "unit-test-vectorizer"


def test_projection_rebuild_only_current_records_and_counts(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-current",
        title="Keep",
        target="target",
        statement="current",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-superseded",
        title="Skip",
        target="target",
        statement="old",
        superseded_by_id="k-current",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-deleted",
        title="Skip",
        target="target",
        statement="removed",
        deleted_at="2026-01-01T00:00:00+00:00",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-other",
        title="Keep2",
        target="target",
        statement="current2",
    )

    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: _SpyVectorizer(_SpyVectorizerConfig()),
    )

    rebuilt = projection.rebuild()
    projection.close()

    assert rebuilt == 2
    store = _read_root(tmp_path / "vectors")
    assert store.ids == ("k-current", "k-other")


def test_projection_rebuild_uses_batches(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")

    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="A",
        target="x",
        statement="1",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-b",
        title="B",
        target="x",
        statement="2",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-c",
        title="C",
        target="x",
        statement="3",
    )

    vectorizer = _SpyVectorizer(_SpyVectorizerConfig())
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: vectorizer,
    )

    rebuilt = projection.rebuild(batch_size=2)
    projection.close()
    assert rebuilt == 3
    assert [len(batch) for batch in vectorizer.calls] == [2, 1]


def test_projection_rebuild_failure_keeps_previous_index(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="First",
        target="target",
        statement="one",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-b",
        title="Second",
        target="target",
        statement="two",
    )

    baseline = _SpyVectorizer(_SpyVectorizerConfig(fingerprint="baseline"))
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: baseline,
    )
    assert projection.rebuild() == 2
    projection.close()

    failing = _FailingSpyVectorizer(_SpyVectorizerConfig(fingerprint="interrupted"), fail_on_call=2)
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: failing,
    )

    with pytest.raises(RuntimeError, match="simulated rebuild interruption"):
        projection.rebuild(batch_size=1)
    projection.close()

    store = _read_root(tmp_path / "vectors")
    assert store.document_count == 2
    assert store.manifest["model_fingerprint"] == "baseline"


def test_projection_rebuild_updates_model_metadata(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="Model",
        target="target",
        statement="version check",
    )

    first = _SpyVectorizer(_SpyVectorizerConfig(dimension=2, fingerprint="old", model_name="model-old"))
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: first,
    )
    assert projection.rebuild() == 1
    projection.close()

    second = _SpyVectorizer(_SpyVectorizerConfig(dimension=3, fingerprint="new", model_name="model-new"))
    projection = KnowledgeVectorProjection(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=lambda: second,
    )
    assert projection.rebuild() == 1
    projection.close()

    store = _read_root(tmp_path / "vectors")
    assert store.manifest["model_fingerprint"] == "new"
    assert store.manifest["model_name"] == "model-new"
    assert store.manifest["dimension"] == 3
    assert store.vectors == ((float(len("Model target version check")), 1.0, 0.0),)
