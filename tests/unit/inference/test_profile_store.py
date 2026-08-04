from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.persistence import rounds_migrations as migrations


def _profile(profile_id: str = "hypothesis-default") -> InferenceProfile:
    return InferenceProfile(
        profile_id=profile_id,
        base_url="https://provider.example/v1",
        model="test-model",
        secret_ref="provider-main",
    )


def _connection(path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    migrations.apply_migrations(connection)
    connection.execute(
        "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("space", "tests", "2026-08-03T00:00:00+00:00", "2026-08-03T00:00:00+00:00"),
    )
    connection.commit()
    return connection


def test_profile_store_upserts_and_lists_without_secret_values(tmp_path) -> None:
    connection = _connection(tmp_path / "local.db")
    try:
        store = InferenceProfileStore(connection)
        store.upsert(_profile())
        assert store.get("hypothesis-default") == _profile()
        assert store.list_ids() == ("hypothesis-default",)
        row = connection.execute("SELECT * FROM inference_profiles").fetchone()
        assert row["secret_ref"] == "provider-main"
        assert "TOP_SECRET" not in " ".join(str(value) for value in row)
    finally:
        connection.close()


def test_profile_store_binds_memory_space_and_records_safe_audit(tmp_path) -> None:
    connection = _connection(tmp_path / "local.db")
    try:
        store = InferenceProfileStore(connection)
        store.upsert(_profile())
        store.bind("space", hypothesis_profile_id="hypothesis-default")
        binding = store.get_binding("space")
        assert binding is not None
        assert binding.hypothesis_profile_id == "hypothesis-default"

        store.record_egress_audit(
            audit_id="audit-1",
            memory_space_id="space",
            profile_id="hypothesis-default",
            operation="hypothesis",
            provider_kind="openai_compatible",
            model="test-model",
            status="success",
            request_bytes=10,
            response_bytes=20,
            attempts=1,
        )
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(egress_audit)")
        ]
        assert "payload" not in columns
        assert (
            connection.execute("SELECT COUNT(*) FROM egress_audit").fetchone()[0] == 1
        )
    finally:
        connection.close()


def test_profile_store_rejects_binding_to_missing_profile(tmp_path) -> None:
    connection = _connection(tmp_path / "local.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            InferenceProfileStore(connection).bind(
                "space", hypothesis_profile_id="missing"
            )
    finally:
        connection.close()
