from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.persistence.raw_round_repository import SQLiteRawRoundRepository

_PAYLOAD = '{"round":{"events":[]},"secret":"[REDACTED]"}'


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


def _insert_round(
    repository: SQLiteRawRoundRepository,
    *,
    retention_expires_at: str | None = None,
) -> str:
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
        payload_json=_PAYLOAD,
        payload_digest="sha256:" + "a" * 64,
        started_at="2026-08-03T00:00:00+00:00",
        completed_at="2026-08-03T00:01:00+00:00",
        retention_expires_at=retention_expires_at,
    )
    return raw_round_id


def test_raw_payload_is_stored_separately_from_metadata(tmp_path: Path) -> None:
    database = tmp_path / "rounds.db"
    connection, repository = _open_database(database)
    try:
        raw_round_id = _insert_round(repository)
        raw_row = connection.execute(
            "SELECT payload_json FROM raw_rounds WHERE raw_round_id = ?",
            (raw_round_id,),
        ).fetchone()
        payload_row = connection.execute(
            """
            SELECT payload_json, payload_bytes, deleted_at
            FROM raw_round_payloads
            WHERE raw_round_id = ?
            """,
            (raw_round_id,),
        ).fetchone()

        assert raw_row["payload_json"] == "{}"
        assert payload_row["payload_json"] == _PAYLOAD
        assert payload_row["payload_bytes"] == len(_PAYLOAD.encode("utf-8"))
        assert payload_row["deleted_at"] is None
        assert repository.get(raw_round_id).payload_json == _PAYLOAD
    finally:
        connection.close()


def test_normalized_round_is_idempotent_by_raw_round_and_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rounds.db"
    connection, repository = _open_database(database)
    try:
        raw_round_id = _insert_round(repository)
        first = repository.store_normalized_round(
            raw_round_id=raw_round_id,
            normalizer_version=1,
            payload_json='{"normalizer_version":1}',
            payload_digest="sha256:" + "b" * 64,
        )
        second = repository.store_normalized_round(
            raw_round_id=raw_round_id,
            normalizer_version=1,
            payload_json='{"normalizer_version":1}',
            payload_digest="sha256:" + "b" * 64,
        )

        assert first.normalized_round_id == second.normalized_round_id
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM normalized_rounds WHERE raw_round_id = ?",
                (raw_round_id,),
            ).fetchone()[0]
            == 1
        )
        with pytest.raises(ValueError, match="normalized round digest changed"):
            repository.store_normalized_round(
                raw_round_id=raw_round_id,
                normalizer_version=1,
                payload_json='{"normalizer_version":1,"changed":true}',
                payload_digest="sha256:" + "c" * 64,
            )
    finally:
        connection.close()


def test_purge_expired_deletes_payload_but_keeps_provenance_and_hypothesis(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rounds.db"
    connection, repository = _open_database(database)
    try:
        raw_round_id = _insert_round(
            repository,
            retention_expires_at="2026-01-01T00:00:00+00:00",
        )
        repository.store_normalized_round(
            raw_round_id=raw_round_id,
            normalizer_version=1,
            payload_json='{"normalized":true}',
            payload_digest="sha256:" + "b" * 64,
        )
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
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
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
        )
        connection.commit()

        assert repository.purge_expired(now="2026-02-01T00:00:00+00:00") == 1
        assert repository.get(raw_round_id) is not None
        assert (
            connection.execute(
                "SELECT payload_json, payload_bytes, deleted_at "
                "FROM raw_round_payloads WHERE raw_round_id = ?",
                (raw_round_id,),
            ).fetchone()["payload_json"]
            == "{}"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM normalized_rounds WHERE raw_round_id = ?",
                (raw_round_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE raw_round_id = ?",
                (raw_round_id,),
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


def test_processing_lease_reclaims_expired_job_and_heartbeat_extends_it(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rounds.db"
    connection, repository = _open_database(database)
    try:
        raw_round_id = _insert_round(repository)
        repository.create_job(
            job_id="job-1",
            raw_round_id=raw_round_id,
            pipeline_version=1,
            normalizer_version=1,
            prompt_version=1,
        )
        connection.commit()

        claimed = repository.claim_ready_job(worker_id="worker-1", lease_seconds=30)
        assert claimed is not None
        assert claimed.status == "processing"
        assert claimed.claimed_by == "worker-1"
        assert claimed.lease_expires_at is not None
        assert claimed.heartbeat_at is not None
        assert claimed.lease_generation == 1
        assert (
            repository.heartbeat(
                claimed.job_id,
                worker_id="worker-1",
                lease_generation=claimed.lease_generation,
                lease_seconds=30,
            )
            is True
        )
        assert (
            repository.heartbeat(
                claimed.job_id,
                worker_id="other-worker",
                lease_generation=claimed.lease_generation,
                lease_seconds=30,
            )
            is False
        )

        connection.execute(
            "UPDATE round_processing_jobs SET lease_expires_at = ? WHERE job_id = ?",
            ("2020-01-01T00:00:00+00:00", claimed.job_id),
        )
        connection.commit()
        reclaimed = repository.claim_ready_job(worker_id="worker-2", lease_seconds=30)

        assert reclaimed is not None
        assert reclaimed.job_id == claimed.job_id
        assert reclaimed.attempts == 2
        assert reclaimed.claimed_by == "worker-2"
        assert reclaimed.lease_generation == 2
        assert (
            repository.finish_job(
                reclaimed.job_id,
                "completed",
                worker_id="worker-1",
                lease_generation=claimed.lease_generation,
            )
            is False
        )
        assert (
            repository.finish_job(
                reclaimed.job_id,
                "completed",
                worker_id="worker-2",
                lease_generation=reclaimed.lease_generation,
            )
            is True
        )
    finally:
        connection.close()


def test_processing_lease_generation_rejects_stale_heartbeat_and_finish(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rounds.db"
    connection, repository = _open_database(database)
    try:
        raw_round_id = _insert_round(repository)
        repository.create_job(
            job_id="job-1",
            raw_round_id=raw_round_id,
            pipeline_version=1,
            normalizer_version=1,
            prompt_version=1,
        )
        connection.commit()

        first = repository.claim_ready_job(worker_id="worker-1", lease_seconds=30)
        assert first is not None
        connection.execute(
            "UPDATE round_processing_jobs SET lease_expires_at = ? WHERE job_id = ?",
            ("2020-01-01T00:00:00+00:00", first.job_id),
        )
        connection.commit()
        second = repository.claim_ready_job(worker_id="worker-2", lease_seconds=30)
        assert second is not None
        assert second.lease_generation == first.lease_generation + 1

        assert (
            repository.heartbeat(
                first.job_id,
                worker_id="worker-1",
                lease_generation=first.lease_generation,
                lease_seconds=30,
            )
            is False
        )
        assert (
            repository.finish_job(
                first.job_id,
                "failed",
                worker_id="worker-1",
                lease_generation=first.lease_generation,
                error="stale worker",
            )
            is False
        )
        current = repository.get_job_for_round(
            raw_round_id,
            pipeline_version=1,
            normalizer_version=1,
            prompt_version=1,
        )
        assert current is not None
        assert current.status == "processing"
        assert current.claimed_by == "worker-2"
        assert current.lease_generation == second.lease_generation
    finally:
        connection.close()
