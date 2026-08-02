"""Integration tests for POST /v1/atoms API endpoint."""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Settings
from ledgermind_local.persistence import migrations, open_sqlite_connection


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
            "statement": "Local SQLite should be the source of truth for stage 4.3.",
            "rationale": "Stable write path.",
            "result": "Implemented minimal ingestion route.",
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


def test_ingest_atom_returns_201_and_request_id(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    client = _build_client(database_path=database)
    response = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token", "X-Request-ID": "req-new"},
        json=_build_request(idempotency_key=_checksum("a")),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["projections_pending"] is True
    assert response.headers["X-Request-ID"] == "req-new"
    assert response.headers["Cache-Control"] == "no-store"


def test_ingest_atom_returns_200_for_duplicate(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    client = _build_client(database_path=database)
    request = _build_request(idempotency_key=_checksum("b"))

    first = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=request,
    )
    second = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=request,
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["atom_id"] == first.json()["atom_id"]
    assert second.json()["knowledge_id"] == first.json()["knowledge_id"]


def test_concurrent_ingest_same_idempotency_key_creates_one_atom(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    request = _build_request(
        idempotency_key=_checksum("concurrent"),
        memory_space_id="space-concurrent",
    )
    barrier = Barrier(2)

    def post_once():
        with _build_client(database_path=database) as client:
            barrier.wait(timeout=5)
            return client.post(
                "/v1/atoms",
                headers={"Authorization": "Bearer test-token"},
                json=request,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: post_once(), range(2)))

    assert sorted(response.status_code for response in responses) == [200, 201]
    payloads = [response.json() for response in responses]
    assert len({payload["atom_id"] for payload in payloads}) == 1
    assert sum(payload["duplicate"] is False for payload in payloads) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM idempotency_results WHERE memory_space_id = ?",
                ("space-concurrent",),
            ).fetchone()[0]
            == 1
        )


def test_ingest_atom_returns_409_for_idempotency_conflict(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    client = _build_client(database_path=database)
    idem = _checksum("c")
    first_request = _build_request(
        idempotency_key=idem,
        memory_space_id="space-a",
    )
    second_request = _build_request(
        idempotency_key=idem,
        memory_space_id="space-a",
    )
    second_request["atom"]["statement"] = "Different statement to force idempotency conflict."

    assert (
        client.post(
            "/v1/atoms",
            headers={"Authorization": "Bearer test-token"},
            json=first_request,
        ).status_code
        == 201
    )

    conflict = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=second_request,
    )

    assert conflict.status_code == 409


def test_ingest_atom_rejects_request_with_extra_field(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    request = _build_request(idempotency_key=_checksum("d"))
    request["unexpected"] = "unexpected field"

    response = _build_client(database_path=database).post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=request,
    )

    assert response.status_code == 422


def test_ingest_atom_rejects_phase_field(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    request = _build_request(idempotency_key=_checksum("e"))
    request["phase"] = "canonical"

    response = _build_client(database_path=database).post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=request,
    )

    assert response.status_code == 422


def test_ingest_atom_isolated_by_memory_space(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    client = _build_client(database_path=database)

    first = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(idempotency_key=_checksum("f"), memory_space_id="memory-a"),
    )
    second = client.post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(idempotency_key=_checksum("0"), memory_space_id="memory-b"),
    )

    assert first.status_code == 201
    assert second.status_code == 201
    with sqlite3.connect(database) as conn:
        rows = conn.execute(
            "SELECT memory_space_id FROM atoms ORDER BY memory_space_id"
        ).fetchall()
    assert [row[0] for row in rows] == ["memory-a", "memory-b"]


def test_ingest_atom_payload_too_large_returns_413(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    request = _build_request(idempotency_key=_checksum("g"))
    request["atom"]["statement"] = "x" * (2 * 1024 * 1024 + 1)

    response = _build_client(database_path=database).post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=request,
    )

    assert response.status_code == 413


def test_ingest_atom_db_error_returns_503_and_writes_nothing(tmp_path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)

    class _BrokenHandler:
        def handle(self, _command):
            raise sqlite3.DatabaseError("database unavailable")

    class _BrokenApplication:
        def __init__(self, handler):
            self._handler = handler

        def build_ingest_atom_handler(self):
            return self._handler

    with sqlite3.connect(database) as connection:
        before_atoms = connection.execute("SELECT COUNT(*) AS total FROM atoms").fetchone()[0]
        before_knowledge = connection.execute(
            "SELECT COUNT(*) AS total FROM knowledge_items"
        ).fetchone()[0]

    response = TestClient(
        create_app(
            application=_BrokenApplication(_BrokenHandler()),
            settings=Settings(database_path=database, api_token="test-token"),
        )
    ).post(
        "/v1/atoms",
        headers={"Authorization": "Bearer test-token"},
        json=_build_request(idempotency_key=_checksum("1")),
    )

    assert response.status_code == 503

    with sqlite3.connect(database) as connection:
        after_atoms = connection.execute("SELECT COUNT(*) AS total FROM atoms").fetchone()[0]
        after_knowledge = connection.execute(
            "SELECT COUNT(*) AS total FROM knowledge_items"
        ).fetchone()[0]

    assert after_atoms == before_atoms
    assert after_knowledge == before_knowledge
