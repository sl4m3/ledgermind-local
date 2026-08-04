from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.persistence.raw_round_repository import SQLiteRawRoundRepository


def _open_database(path: Path) -> tuple[sqlite3.Connection, SQLiteRawRoundRepository]:
    connection = open_sqlite_connection(path)
    migrations.apply_migrations(connection)
    connection.execute(
        """
        INSERT INTO memory_spaces (
            memory_space_id, display_name, source_client, created_at, updated_at
        ) VALUES ('space-1', 'Test space', 'test', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """
    )
    connection.commit()
    return connection, SQLiteRawRoundRepository(connection)


def _insert_round(repository: SQLiteRawRoundRepository) -> str:
    raw_round_id = "raw-round-1"
    repository.insert(
        raw_round_id=raw_round_id,
        memory_space_id="space-1",
        source_system="hermes",
        source_instance_id="instance-1",
        source_profile_id="profile-1",
        source_session_id="session-1",
        source_round_id="round-1",
        source_round_key="source-key-1",
        capture_schema_version=1,
        adapter_version="test/1",
        payload_json='{"round":{"events":[]}}',
        payload_digest="sha256:" + "a" * 64,
        started_at="2026-08-03T00:00:00+00:00",
        completed_at="2026-08-03T00:01:00+00:00",
    )
    return raw_round_id


def test_core_command_is_idempotent_and_rejects_payload_conflict(
    tmp_path: Path,
) -> None:
    connection, repository = _open_database(tmp_path / "rounds.db")
    try:
        first = repository.create_core_command(
            command_id="command-1",
            command_type="accept_hypothesis",
            memory_space_id="space-1",
            idempotency_key="sha256:" + "b" * 64,
            payload_json='{"hypothesis_id":"hypothesis-1"}',
            payload_digest="sha256:" + "c" * 64,
            available_at="2026-08-03T00:00:00+00:00",
        )
        duplicate = repository.create_core_command(
            command_id="different-command-id",
            command_type="accept_hypothesis",
            memory_space_id="space-1",
            idempotency_key="sha256:" + "b" * 64,
            payload_json='{"hypothesis_id":"hypothesis-1"}',
            payload_digest="sha256:" + "c" * 64,
            available_at="2026-08-03T00:00:00+00:00",
        )

        assert duplicate.command_id == first.command_id
        assert duplicate.status == "pending"
        assert (
            connection.execute("SELECT COUNT(*) FROM core_commands").fetchone()[0] == 1
        )

        with pytest.raises(ValueError, match="core command payload conflict"):
            repository.create_core_command(
                command_id="command-2",
                command_type="accept_hypothesis",
                memory_space_id="space-1",
                idempotency_key="sha256:" + "b" * 64,
                payload_json='{"hypothesis_id":"other"}',
                payload_digest="sha256:" + "d" * 64,
                available_at="2026-08-03T00:00:00+00:00",
            )
    finally:
        connection.close()


def test_core_command_lease_reclaims_and_finishes_by_owner(tmp_path: Path) -> None:
    connection, repository = _open_database(tmp_path / "rounds.db")
    try:
        repository.create_core_command(
            command_id="command-1",
            command_type="accept_hypothesis",
            memory_space_id="space-1",
            idempotency_key="sha256:" + "b" * 64,
            payload_json='{"hypothesis_id":"hypothesis-1"}',
            payload_digest="sha256:" + "c" * 64,
            available_at="2026-08-03T00:00:00+00:00",
        )
        connection.commit()

        first = repository.claim_core_command(
            worker_id="core-worker-1",
            lease_seconds=30,
            now="2026-08-03T00:00:00+00:00",
        )
        assert first is not None
        assert first.status == "delivering"
        assert first.attempts == 1
        assert first.claimed_by == "core-worker-1"

        connection.execute(
            "UPDATE core_commands SET lease_expires_at = ? WHERE command_id = ?",
            ("2020-01-01T00:00:00+00:00", first.command_id),
        )
        connection.commit()
        reclaimed = repository.claim_core_command(
            worker_id="core-worker-2",
            lease_seconds=30,
            now="2026-08-03T00:00:01+00:00",
        )
        assert reclaimed is not None
        assert reclaimed.attempts == 2
        assert reclaimed.claimed_by == "core-worker-2"

        assert (
            repository.finish_core_command(
                reclaimed.command_id,
                worker_id="core-worker-1",
                status="completed",
                result_json='{"accepted":true}',
            )
            is False
        )
        assert (
            repository.finish_core_command(
                reclaimed.command_id,
                worker_id="core-worker-2",
                status="completed",
                result_json='{"accepted":true}',
            )
            is True
        )
        finished = repository.get_core_command(reclaimed.command_id)
        assert finished is not None
        assert finished.status == "completed"
        assert finished.result_json == '{"accepted":true}'
    finally:
        connection.close()


def test_core_command_payload_and_hypothesis_status_are_persisted_in_processing_schema(
    tmp_path: Path,
) -> None:
    connection, repository = _open_database(tmp_path / "rounds.db")
    try:
        raw_round_id = _insert_round(repository)
        repository.create_job(
            job_id="job-1",
            raw_round_id=raw_round_id,
            pipeline_version=1,
            normalizer_version=1,
            prompt_version=1,
        )
        repository.create_attempt(
            attempt_id="attempt-1",
            raw_round_id=raw_round_id,
            job_id="job-1",
            pipeline_version=1,
            normalizer_version=1,
            provider="test",
            model="test",
            prompt_version=1,
            schema_version=1,
            started_at="2026-08-03T00:00:00+00:00",
            completed_at="2026-08-03T00:00:01+00:00",
        )
        repository.insert_hypothesis(
            hypothesis_id="hypothesis-1",
            raw_round_id=raw_round_id,
            attempt_id="attempt-1",
            hypothesis_index=0,
            title="Title",
            target="Target",
            statement="Statement",
            rationale="Rationale",
            result="Result",
            artifacts_json="[]",
            content_digest="sha256:" + "d" * 64,
            status="queued_for_core",
            core_command_id="command-1",
        )
        connection.commit()

        hypothesis = connection.execute(
            "SELECT status, core_command_id FROM hypotheses WHERE hypothesis_id = ?",
            ("hypothesis-1",),
        ).fetchone()
        assert tuple(hypothesis) == ("queued_for_core", "command-1")
    finally:
        connection.close()
