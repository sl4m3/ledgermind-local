"""Integration tests for reading atoms and knowledge by memory space."""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import Settings
from persistence import migrations, open_sqlite_connection


def _checksum(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bootstrap_database(database_path) -> None:
    connection = open_sqlite_connection(database_path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _build_request(*, idempotency_key: str, memory_space_id: str = "space-a") -> dict:
    return {
        "api_version": "1",
        "idempotency_key": idempotency_key,
        "memory_space_id": memory_space_id,
        "source": {
            "source_system": "hermes",
            "source_instance_id": "instance-1",
            "source_profile_id": "profile-1",
            "source_session_id": "session-1",
            "source_round_id": "round-1",
            "first_message_id": "m-1",
            "final_message_id": "m-3",
            "message_ids": ["m-1", "m-2"],
            "source_digest": "sha256:" + "a" * 64,
            "source_schema_version": 1,
            "resolver_version": 1,
        },
        "extraction": {
            "host": "hermes",
            "provider": "openrouter",
            "model": "gpt-4o",
            "prompt_version": 1,
            "schema_version": 1,
            "purpose": "ledgermind.atom.extract",
        },
        "atom": {
            "title": "How to keep canonical memory",
            "target": "architecture.persistence",
            "statement": "Local SQLite should be the source of truth for stage 4.4.",
            "rationale": "Stable read path.",
            "result": "Need atom and knowledge reads.",
            "artifacts": ["artifact-1", "artifact-2"],
        },
    }


def _build_client(*, database_path, token: str = "test-token") -> TestClient:
    return TestClient(
        create_app(
            application=object(),
            settings=Settings(database_path=database_path, api_token=token),
        )
    )


def _create_atom(database_path, *, memory_space_id: str) -> dict:
    response = _build_client(database_path=database_path).post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(
            idempotency_key=_checksum("k" + memory_space_id),
            memory_space_id=memory_space_id,
        ),
    )
    assert response.status_code == 201
    return response.json()


def test_get_atom_requires_token(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    memory_space_id = "space-a"
    payload = _create_atom(database, memory_space_id=memory_space_id)

    response = _build_client(database_path=database, token="test-token").get(
        f"/v1/atoms/{memory_space_id}/{payload['atom_id']}"
    )
    assert response.status_code == 401


def test_get_atom_returns_200_with_space_and_atom_id(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    memory_space_id = "space-a"
    payload = _create_atom(database, memory_space_id=memory_space_id)

    response = _build_client(database_path=database).get(
        f"/v1/atoms/{memory_space_id}/{payload['atom_id']}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "1"
    assert body["atom_id"] == payload["atom_id"]
    assert body["memory_space_id"] == "space-a"
    assert body["source"]["source_round_id"] == "round-1"
    assert body["extraction"]["host"] == "hermes"


def test_get_atom_isolation_requires_space(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    payload = _create_atom(database, memory_space_id="space-a")

    response = _build_client(database_path=database).get(
        f"/v1/atoms/space-b/{payload['atom_id']}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404


def test_get_atom_not_found_for_other_space(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    payload = _create_atom(database, memory_space_id="space-a")

    response = _build_client(database_path=database).get(
        f"/v1/atoms/space-b/{payload['atom_id']}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404


def test_get_atom_returns_404_for_unknown_atom(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    response = _build_client(database_path=database).get(
        "/v1/atoms/space-a/unknown-atom",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404


def test_get_knowledge_returns_200_with_space_and_id(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    memory_space_id = "space-a"
    payload = _create_atom(database, memory_space_id=memory_space_id)

    response = _build_client(database_path=database).get(
        f"/v1/knowledge/{memory_space_id}/{payload['knowledge_id']}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "1"
    assert body["knowledge_id"] == payload["knowledge_id"]
    assert body["memory_space_id"] == "space-a"
    assert body["title"] == "How to keep canonical memory"
    assert body["phase"] == "pattern"


def test_get_knowledge_not_found_for_unknown_id(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    response = _build_client(database_path=database).get(
        "/v1/knowledge/space-a/unknown-knowledge",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404


def test_get_knowledge_history_returns_items(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    memory_space_id = "space-a"
    payload = _create_atom(database, memory_space_id=memory_space_id)

    response = _build_client(database_path=database).get(
        f"/v1/memory-spaces/{memory_space_id}/knowledge/{payload['knowledge_id']}/history",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "1"
    assert body["knowledge_id"] == payload["knowledge_id"]
    assert body["memory_space_id"] == memory_space_id
    assert isinstance(body["revisions"], list)
    assert len(body["revisions"]) >= 1
    first = body["revisions"][0]
    assert isinstance(first["snapshot"], dict)
    assert first["version"] == 1


def test_get_knowledge_evidence_returns_items(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    memory_space_id = "space-a"
    payload = _create_atom(database, memory_space_id=memory_space_id)

    response = _build_client(database_path=database).get(
        f"/v1/memory-spaces/{memory_space_id}/knowledge/{payload['knowledge_id']}/evidence",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "1"
    assert body["knowledge_id"] == payload["knowledge_id"]
    assert body["memory_space_id"] == memory_space_id
    assert isinstance(body["evidence"], list)
    assert len(body["evidence"]) >= 1
    assert body["evidence"][0]["atom_id"] == payload["atom_id"]


def test_get_knowledge_history_not_found_for_unknown_knowledge(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    response = _build_client(database_path=database).get(
        "/v1/memory-spaces/space-a/knowledge/missing/history",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404


def test_get_knowledge_evidence_not_found_for_unknown_knowledge(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    response = _build_client(database_path=database).get(
        "/v1/memory-spaces/space-a/knowledge/missing/evidence",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404
