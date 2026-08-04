from __future__ import annotations

import sqlite3
from pathlib import Path

from ledgermind_local.core_gateway.contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    DomainRejectedError,
    HypothesisEvidence,
    HypothesisExtraction,
    HypothesisPayload,
    TransientCoreError,
)
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.persistence.raw_round_repository import SQLiteRawRoundRepository
from ledgermind_local.scheduler.core_command_worker import CoreCommandWorker


def _bootstrap(path: Path) -> tuple[sqlite3.Connection, SQLiteRawRoundRepository]:
    connection = open_sqlite_connection(path)
    migrations.apply_migrations(connection)
    connection.execute(
        """
        INSERT INTO memory_spaces (
            memory_space_id, display_name, source_client, created_at, updated_at
        ) VALUES ('space-1', 'Test space', 'test', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """
    )
    repository = SQLiteRawRoundRepository(connection)
    repository.insert(
        raw_round_id="raw-round-1",
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
    repository.create_job(
        job_id="job-1",
        raw_round_id="raw-round-1",
        pipeline_version=1,
        normalizer_version=1,
        prompt_version=1,
    )
    repository.create_attempt(
        attempt_id="attempt-1",
        raw_round_id="raw-round-1",
        job_id="job-1",
        pipeline_version=1,
        normalizer_version=1,
        provider="test-provider",
        model="test-model",
        prompt_version=1,
        schema_version=1,
        started_at="2026-08-03T00:00:00+00:00",
        completed_at="2026-08-03T00:00:01+00:00",
    )
    repository.insert_hypothesis(
        hypothesis_id="hypothesis-1",
        raw_round_id="raw-round-1",
        attempt_id="attempt-1",
        hypothesis_index=0,
        title="Deployment rule",
        target="operations",
        statement="Use staging first.",
        rationale="Safer rollout.",
        result="Staging passed.",
        artifacts_json="[]",
        content_digest="sha256:" + "b" * 64,
        status="queued_for_core",
        core_command_id="command-1",
    )
    command = _command()
    repository.create_core_command(
        command_id=command.command_id,
        command_type="accept_hypothesis",
        memory_space_id=command.memory_space_id,
        idempotency_key=command.idempotency_key,
        payload_json=command.to_json(),
        payload_digest="sha256:" + "c" * 64,
        available_at="2026-08-03T00:00:00+00:00",
    )
    connection.commit()
    return connection, repository


def _command() -> AcceptHypothesisCommand:
    return AcceptHypothesisCommand(
        protocol_version=1,
        command_id="command-1",
        idempotency_key="sha256:" + "d" * 64,
        memory_space_id="space-1",
        hypothesis=HypothesisPayload(
            hypothesis_id="hypothesis-1",
            content_digest="sha256:" + "b" * 64,
            title="Deployment rule",
            target="operations",
            statement="Use staging first.",
            rationale="Safer rollout.",
            result="Staging passed.",
            artifacts=(),
            evidence=HypothesisEvidence(
                source_system="hermes",
                source_instance_id="instance-1",
                source_profile_id="profile-1",
                source_session_id="session-1",
                source_round_id="round-1",
                raw_round_digest="sha256:" + "a" * 64,
                normalized_round_digest="sha256:" + "e" * 64,
                source_event_ids=("event-1",),
            ),
            extraction=HypothesisExtraction(
                provider="test-provider",
                model="test-model",
                prompt_version=1,
                schema_version=1,
                completed_at="2026-08-03T00:00:01+00:00",
            ),
        ),
    )


class _Gateway:
    def __init__(self, outcomes=()) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def accept_hypothesis(
        self, command: AcceptHypothesisCommand
    ) -> AcceptHypothesisResult:
        self.calls.append(command.command_id)
        outcome = (
            self.outcomes.pop(0)
            if self.outcomes
            else AcceptHypothesisResult(
                accepted=True,
                duplicate=False,
                core_reference_id="knowledge-1",
                result_json='{"knowledge_id":"knowledge-1"}',
            )
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_core_command_worker_completes_and_does_not_redeliver_completed_command(
    tmp_path: Path,
) -> None:
    connection, _ = _bootstrap(tmp_path / "rounds.db")
    connection.close()
    gateway = _Gateway()
    worker = CoreCommandWorker(
        database_path=tmp_path / "rounds.db",
        gateway=gateway,
        worker_id="core-worker-1",
    )

    first = worker.process_once()
    second = worker.process_once()

    assert first is not None
    assert first.status == "completed"
    assert second is None
    assert gateway.calls == ["command-1"]
    with sqlite3.connect(tmp_path / "rounds.db") as check:
        assert check.execute(
            "SELECT status, result_json FROM core_commands"
        ).fetchone() == ("completed", '{"knowledge_id":"knowledge-1"}')
        assert check.execute(
            "SELECT status FROM hypotheses WHERE hypothesis_id = 'hypothesis-1'"
        ).fetchone() == ("accepted_by_core",)


def test_core_command_worker_retries_transient_error_and_then_completes(
    tmp_path: Path,
) -> None:
    connection, _ = _bootstrap(tmp_path / "rounds.db")
    connection.close()
    gateway = _Gateway(
        [
            TransientCoreError("core unavailable"),
            AcceptHypothesisResult(
                accepted=True,
                duplicate=True,
                core_reference_id="knowledge-1",
                result_json='{"duplicate":true}',
            ),
        ]
    )
    worker = CoreCommandWorker(
        database_path=tmp_path / "rounds.db",
        gateway=gateway,
        worker_id="core-worker-1",
        retry_delay_seconds=0,
    )

    failed = worker.process_once()
    completed = worker.process_once()

    assert failed is not None
    assert failed.status == "retry_wait"
    assert completed is not None
    assert completed.status == "completed"
    assert gateway.calls == ["command-1", "command-1"]
    with sqlite3.connect(tmp_path / "rounds.db") as check:
        assert check.execute(
            "SELECT attempts, status FROM core_commands"
        ).fetchone() == (2, "completed")


def test_core_command_worker_rejects_domain_error_without_infinite_retry(
    tmp_path: Path,
) -> None:
    connection, _ = _bootstrap(tmp_path / "rounds.db")
    connection.close()
    gateway = _Gateway([DomainRejectedError("invalid_hypothesis", "content rejected")])
    worker = CoreCommandWorker(
        database_path=tmp_path / "rounds.db",
        gateway=gateway,
        worker_id="core-worker-1",
    )

    rejected = worker.process_once()
    again = worker.process_once()

    assert rejected is not None
    assert rejected.status == "rejected"
    assert again is None
    assert gateway.calls == ["command-1"]
    with sqlite3.connect(tmp_path / "rounds.db") as check:
        assert check.execute(
            "SELECT status, last_error_code FROM core_commands"
        ).fetchone() == ("rejected", "invalid_hypothesis")
        assert check.execute(
            "SELECT status FROM hypotheses WHERE hypothesis_id = 'hypothesis-1'"
        ).fetchone() == ("rejected_by_core",)


def test_core_command_worker_reclaims_expired_delivery_after_core_acceptance(
    tmp_path: Path,
) -> None:
    connection, repository = _bootstrap(tmp_path / "rounds.db")
    claimed = repository.claim_core_command(
        worker_id="crashed-worker",
        lease_seconds=30,
        now="2026-08-03T00:00:00+00:00",
    )
    assert claimed is not None
    connection.execute(
        "UPDATE core_commands SET lease_expires_at = ? WHERE command_id = ?",
        ("2020-01-01T00:00:00+00:00", claimed.command_id),
    )
    connection.commit()
    connection.close()

    gateway = _Gateway(
        [
            AcceptHypothesisResult(
                accepted=True,
                duplicate=True,
                core_reference_id="knowledge-1",
                result_json='{"duplicate":true}',
            )
        ]
    )
    worker = CoreCommandWorker(
        database_path=tmp_path / "rounds.db",
        gateway=gateway,
        worker_id="recovery-worker",
    )

    result = worker.process_once()

    assert result is not None
    assert result.status == "completed"
    assert gateway.calls == ["command-1"]
