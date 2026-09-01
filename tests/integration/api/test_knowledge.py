from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ledgermind_local.api.knowledge import create_knowledge_router


class _Gateway:
    def __init__(self) -> None:
        self.memory_space_ids: list[str] = []
        self.deleted_memory_space_ids: list[str] = []

    def get_object_facet_snapshot(
        self, memory_space_id: str, request_id: str
    ) -> dict[str, Any]:
        assert request_id
        self.memory_space_ids.append(memory_space_id)
        return {
            "schema_version": 1,
            "memory_space_id": memory_space_id,
            "knowledge_values": [{"value_id": f"value:{memory_space_id}"}],
        }

    def delete_memory_space(self, memory_space_id: str, request_id: str) -> bool:
        assert request_id
        self.deleted_memory_space_ids.append(memory_space_id)
        return True


def test_export_contains_only_matching_memory_spaces(tmp_path: Path) -> None:
    database_path = tmp_path / "rounds.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE memory_spaces (memory_space_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO memory_spaces VALUES (?)",
            [
                ("tenant:user:workspace:one:default",),
                ("tenant:user:workspace:one:project",),
                ("tenant:user:workspace:two:default",),
            ],
        )

    gateway = _Gateway()
    app = FastAPI()
    app.include_router(
        create_knowledge_router(
            lambda: "token", gateway, database_path=database_path
        )
    )

    response = TestClient(app).post(
        "/knowledge/export",
        json={"memory_space_prefix": "tenant:user:workspace:one:"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert [item["memory_space_id"] for item in payload["memory_spaces"]] == [
        "default",
        "project",
    ]
    assert gateway.memory_space_ids == [
        "tenant:user:workspace:one:default",
        "tenant:user:workspace:one:project",
    ]

    purged = TestClient(app).post(
        "/knowledge/purge",
        json={"memory_space_prefix": "tenant:user:workspace:one:"},
    )
    assert purged.status_code == 200
    assert purged.json() == {"deleted_memory_spaces": 2}
    assert gateway.deleted_memory_space_ids == [
        "tenant:user:workspace:one:default",
        "tenant:user:workspace:one:project",
    ]
    with sqlite3.connect(database_path) as connection:
        remaining = connection.execute(
            "SELECT memory_space_id FROM memory_spaces"
        ).fetchall()
    assert remaining == [("tenant:user:workspace:two:default",)]
