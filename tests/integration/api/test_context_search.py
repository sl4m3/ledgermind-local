"""Integration tests for POST /v1/context/search API endpoint."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from domain.events import KnowledgeCreated
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import Settings
from persistence import (
    Atom,
    Knowledge,
    KnowledgeEvidence,
    SQLiteAtomRepository,
    SQLiteEvidenceRepository,
    SQLiteKnowledgeRepository,
    migrations,
    open_sqlite_connection,
)
from projections import KnowledgeFTSProjection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bootstrap_database(database_path) -> None:
    connection = open_sqlite_connection(database_path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _build_connection(database_path):
    connection = open_sqlite_connection(database_path)
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


def _add_atom(
    connection,
    *,
    atom_id: str,
    memory_space_id: str,
    source_round_id: str = "round-1",
) -> None:
    SQLiteAtomRepository(connection).add(
        Atom(
            atom_id=atom_id,
            memory_space_id=memory_space_id,
            source_system="hermes",
            source_instance_id="instance-1",
            source_profile_id="profile-1",
            source_session_id="session-1",
            source_round_id=source_round_id,
            source_round_key=f"{memory_space_id}:{source_round_id}",
            first_message_id="m-1",
            final_message_id="m-3",
            message_ids=("m-1", "m-2"),
            source_digest="sha256:" + "a" * 64,
            source_schema_version=1,
            resolver_version=1,
            extraction_host="hermes",
            extraction_provider="openrouter",
            extraction_model="gpt-4o",
            extraction_prompt_version=1,
            extraction_schema_version=1,
            extraction_purpose="ledgermind.atom.extract",
            title="How to keep context search stable",
            target="engineering",
            statement="Atom statement.",
            rationale="Atom rationale",
            result="Atom result",
            artifacts=("artifact-1",),
            content_digest="sha256:" + "b" * 64,
            supersedes_atom_id=None,
            created_at=_now(),
        )
    )


def _add_knowledge(
    connection,
    *,
    knowledge_id: str,
    memory_space_id: str,
    title: str,
    target: str,
    statement: str,
    phase: str = "pattern",
) -> None:
    SQLiteKnowledgeRepository(connection).add(
        Knowledge(
            knowledge_id=knowledge_id,
            memory_space_id=memory_space_id,
            title=title,
            target=target,
            statement=statement,
            rationale="",
            phase=phase,
            version=1,
            created_at=_now(),
            updated_at=_now(),
            superseded_by_id=None,
            deleted_at=None,
        )
    )


def _add_evidence(
    connection,
    *,
    knowledge_id: str,
    atom_id: str,
) -> None:
    SQLiteEvidenceRepository(connection).add(
        KnowledgeEvidence(
            knowledge_id=knowledge_id,
            atom_id=atom_id,
            relation="origin",
            created_at=_now(),
        )
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


def _build_client(*, database_path, token: str = "test-token") -> TestClient:
    return TestClient(
        create_app(
            application=object(),
            settings=Settings(database_path=database_path, api_token=token),
        )
    )


def _build_request(
    *,
    memory_space_id: str,
    query: str,
    limit: int = 5,
    min_phase: str | None = None,
) -> dict[str, str | int | None]:
    request = {
        "api_version": "1",
        "memory_space_id": memory_space_id,
        "query": query,
        "limit": limit,
    }
    if min_phase is not None:
        request["min_phase"] = min_phase
    return request


def _assert_context_item_fields(item: dict[str, object]) -> None:
    assert isinstance(item["knowledge_id"], str)
    assert isinstance(item["title"], str)
    assert isinstance(item["target"], str)
    assert isinstance(item["statement"], str)
    assert isinstance(item["rationale"], str)
    assert item["phase"] in {"pattern", "emergent", "canonical"}
    assert isinstance(item["score"], int | float)
    assert 0.0 <= float(item["score"]) <= 1.0
    assert isinstance(item["evidence_count"], int)
    assert isinstance(item["source_atom_ids"], list)
    assert "source_reference" not in item


def test_context_search_returns_items_after_projection_is_processed(tmp_path) -> None:
    database = tmp_path / "state.db"
    connection = _build_connection(database)
    _ensure_space(connection, "space-a")
    _add_atom(connection, atom_id="atom-a", memory_space_id="space-a")
    _add_knowledge(
        connection,
        knowledge_id="k-canonical",
        memory_space_id="space-a",
        title="Canonical knowledge",
        target="architecture",
        statement="Context search should return canonical pattern after projection.",
    )
    _add_evidence(connection, knowledge_id="k-canonical", atom_id="atom-a")
    _add_knowledge(
        connection,
        knowledge_id="k-irrelevant",
        memory_space_id="space-a",
        title="Different knowledge",
        target="architecture",
        statement="No related phrase here.",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-canonical")
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-irrelevant")
    connection.commit()
    connection.close()

    response = _build_client(database_path=database).post(
        "/v1/context/search",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            memory_space_id="space-a",
            query="context canonical pattern",
            limit=5,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "1"
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["knowledge_id"] == "k-canonical"
    assert item["score"] >= 0.0
    assert item["evidence_count"] == 1
    assert item["source_atom_ids"] == ["atom-a"]
    _assert_context_item_fields(item)


def test_context_search_works_without_vector_data(tmp_path) -> None:
    database = tmp_path / "state.db"
    connection = _build_connection(database)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        knowledge_id="k-vectorless",
        memory_space_id="space-a",
        title="Vectorless fallback",
        target="engineering",
        statement="Search should still work when vector score is absent.",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-vectorless")
    connection.commit()
    connection.close()

    response = _build_client(database_path=database).post(
        "/v1/context/search",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            memory_space_id="space-a",
            query="search fallback",
            limit=3,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["knowledge_id"] == "k-vectorless"
    assert item["evidence_count"] == 0
    assert item["source_atom_ids"] == []
    _assert_context_item_fields(item)


def test_context_search_respects_limit(tmp_path) -> None:
    database = tmp_path / "state.db"
    connection = _build_connection(database)
    _ensure_space(connection, "space-a")
    projection = KnowledgeFTSProjection(connection)

    for number in range(1, 5):
        knowledge_id = f"k-{number:03d}"
        _add_knowledge(
            connection,
            knowledge_id=knowledge_id,
            memory_space_id="space-a",
            title="Shared anchor",
            target="engineering",
            statement="Shared anchor phrase for ranking.",
        )
        _emit_created(projection, memory_space_id="space-a", knowledge_id=knowledge_id)

    connection.commit()
    connection.close()

    response = _build_client(database_path=database).post(
        "/v1/context/search",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            memory_space_id="space-a",
            query="shared anchor",
            limit=2,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["knowledge_id"] for item in body["items"]] == ["k-001", "k-002"]
    assert len(body["items"]) == 2


def test_context_search_isolated_by_memory_space(tmp_path) -> None:
    database = tmp_path / "state.db"
    connection = _build_connection(database)
    _ensure_space(connection, "space-a")
    _ensure_space(connection, "space-b")
    _add_knowledge(
        connection,
        knowledge_id="k-a",
        memory_space_id="space-a",
        title="Space A anchor",
        target="architecture",
        statement="shared anchor in space A",
    )
    _add_knowledge(
        connection,
        knowledge_id="k-b",
        memory_space_id="space-b",
        title="Space B anchor",
        target="architecture",
        statement="shared anchor in space B",
    )

    projection = KnowledgeFTSProjection(connection)
    _emit_created(projection, memory_space_id="space-a", knowledge_id="k-a")
    _emit_created(projection, memory_space_id="space-b", knowledge_id="k-b")
    connection.commit()
    connection.close()

    response = _build_client(database_path=database).post(
        "/v1/context/search",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            memory_space_id="space-a",
            query="shared anchor",
            limit=10,
        ),
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["knowledge_id"] == "k-a"


def test_context_search_returns_503_for_database_errors(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    class _BrokenHandler:
        def handle(self, _query):
            raise sqlite3.DatabaseError("database unavailable")

    class _BrokenApplication:
        def __init__(self, handler):
            self._handler = handler

        def build_retrieve_context_handler(self):
            return self._handler

    response = TestClient(
        create_app(
            application=_BrokenApplication(_BrokenHandler()),
            settings=Settings(database_path=database, api_token="test-token"),
        )
    ).post(
        "/v1/context/search",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            memory_space_id="space-a",
            query="anything",
            limit=3,
        ),
    )

    assert response.status_code == 503
