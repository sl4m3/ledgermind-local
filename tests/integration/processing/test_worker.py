from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ledgermind_protocol import RawRoundRequest, calculate_raw_round_digest

from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.persistence.raw_round_repository import SQLiteRawRoundRepository
from ledgermind_local.processing.generator import HypothesisCandidate
from ledgermind_local.processing.models import NormalizedRound
from ledgermind_local.processing.normalizer import normalize_raw_round
from ledgermind_local.processing.worker import RoundProcessingWorker
from ledgermind_local.raw_rounds import RawRoundIngestHandler


def _bootstrap(path: Path) -> None:
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _request() -> RawRoundRequest:
    payload = {
        "api_version": "2",
        "idempotency_key": "sha256:" + "a" * 64,
        "memory_space_id": "space",
        "source": {
            "system": "hermes",
            "instance_id": "instance",
            "profile_id": "profile",
            "session_id": "session",
            "round_id": "round",
            "first_event_id": "m-1",
            "final_event_id": "m-2",
            "event_ids": ["m-1", "m-2"],
            "source_schema_version": 1,
            "adapter_version": "test/1",
        },
        "round": {
            "started_at": "2026-08-02T20:00:00Z",
            "completed_at": "2026-08-02T20:01:00Z",
            "events": [
                {
                    "event_id": "m-1",
                    "sequence": 0,
                    "kind": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "request"}],
                },
                {
                    "event_id": "m-2",
                    "sequence": 1,
                    "kind": "message",
                    "role": "assistant",
                    "final": True,
                    "content": [{"type": "text", "text": "answer"}],
                },
            ],
        },
        "payload_digest": "sha256:" + "0" * 64,
    }
    payload["payload_digest"] = calculate_raw_round_digest(payload)
    payload["idempotency_key"] = payload["payload_digest"]
    return RawRoundRequest.model_validate(payload)


class _Generator:
    provider = "test-provider"
    model = "test-model"
    prompt_version = 2
    schema_version = 3

    def __init__(
        self,
        drafts: tuple[HypothesisCandidate, ...] = (),
        error: Exception | None = None,
        probe=None,
    ) -> None:
        self.drafts = tuple(drafts)
        self.error = error
        self.probe = probe

    def generate(
        self, normalized_round: NormalizedRound
    ) -> tuple[HypothesisCandidate, ...]:
        del normalized_round
        if self.probe is not None:
            self.probe()
        if self.error is not None:
            raise self.error
        return self.drafts


def _ingest(path: Path) -> None:
    result = RawRoundIngestHandler(database_path=path).handle(_request())
    assert result.duplicate is False


def _reclaim_stale_worker(database: Path, drafts=()):
    worker = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(drafts=drafts),
        worker_id="worker-a",
        lease_seconds=30,
    )
    claimed = worker._claim()
    assert claimed is not None
    job, raw_round = claimed
    connection = open_sqlite_connection(database)
    try:
        connection.execute(
            "UPDATE round_processing_jobs SET lease_expires_at = ? WHERE job_id = ?",
            ("2020-01-01T00:00:00+00:00", job.job_id),
        )
        reclaimed = SQLiteRawRoundRepository(connection).claim_ready_job(
            worker_id="worker-b",
            lease_seconds=30,
        )
        assert reclaimed is not None
        assert reclaimed.lease_generation == job.lease_generation + 1
        connection.commit()
    finally:
        connection.close()
    normalized = normalize_raw_round(
        json.loads(raw_round.payload_json),
        normalizer_version=job.normalizer_version,
    )
    return worker, job, raw_round, normalized


def test_worker_persists_hypothesis_and_core_command_outside_write_lock(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    _ingest(database)
    lock_probe = {"ok": False}

    def probe_lock() -> None:
        connection = sqlite3.connect(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            lock_probe["ok"] = True
            connection.rollback()
        finally:
            connection.close()

    worker = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(
            drafts=(
                HypothesisCandidate(
                    title="Deployment rule",
                    target="operations",
                    statement="Deployments require a staging check token=secret.",
                    artifacts=("runbook",),
                ),
            ),
            probe=probe_lock,
        ),
        worker_id="worker-1",
    )

    result = worker.process_once()

    assert result is not None
    assert result.status == "completed"
    assert lock_probe["ok"] is True
    assert result.hypothesis_ids
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT status FROM round_processing_jobs").fetchone()
        assert row == ("completed",)
        hypothesis = connection.execute(
            "SELECT title, statement, artifacts_json, status, core_command_id FROM hypotheses"
        ).fetchone()
        assert hypothesis[0] == "Deployment rule"
        assert "secret" not in hypothesis[1]
        assert "[REDACTED]" in hypothesis[1]
        assert hypothesis[3] == "queued_for_core"
        assert hypothesis[4]
        assert connection.execute("SELECT COUNT(*) FROM raw_rounds").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0]
            == 1
        )
        assert connection.execute(
            "SELECT command_type, status FROM core_commands"
        ).fetchone() == ("accept_hypothesis", "pending")


def test_worker_marks_no_knowledge_without_hypothesis_command(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    _ingest(database)
    result = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(),
    ).process_once()

    assert result is not None
    assert result.status == "no_knowledge"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM core_commands").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0]
            == 1
        )


def test_worker_retries_generation_error_without_duplicate_raw_round(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    _ingest(database)

    result = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(error=RuntimeError("provider token=secret failed")),
        retry_delay_seconds=0,
        max_attempts=3,
    ).process_once()

    assert result is not None
    assert result.status == "retry_wait"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_rounds").fetchone()[0] == 1
        assert connection.execute(
            "SELECT attempts, status FROM round_processing_jobs"
        ).fetchone() == (1, "retry_wait")
        error_detail = connection.execute(
            "SELECT error_detail FROM hypothesis_attempts"
        ).fetchone()[0]
        assert "secret" not in error_detail
        assert "[REDACTED]" in error_detail


def test_stale_worker_success_rolls_back_attempt_hypotheses_and_commands(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stale-success.db"
    _bootstrap(database)
    _ingest(database)
    worker, job, raw_round, normalized = _reclaim_stale_worker(
        database,
        drafts=(_candidate("Alpha"),),
    )

    result = worker._persist_success(
        job,
        raw_round,
        (_candidate("Alpha"),),
        normalized=normalized,
        started_at="2026-08-04T00:00:00+00:00",
    )

    assert result.status == "lease_lost"
    assert result.hypothesis_ids == ()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT status, claimed_by, lease_generation, attempts "
            "FROM round_processing_jobs"
        ).fetchone() == ("processing", "worker-b", 2, 2)
        assert connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM core_commands").fetchone()[0] == 0


def test_stale_worker_error_rolls_back_attempt_before_retry_or_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stale-error.db"
    _bootstrap(database)
    _ingest(database)
    worker, job, raw_round, _ = _reclaim_stale_worker(database)

    result = worker._error_result(
        job,
        raw_round,
        error_code="processing_error",
        error=RuntimeError("provider token=secret failed"),
        started_at="2026-08-04T00:00:00+00:00",
    )

    assert result.status == "lease_lost"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT status, claimed_by, lease_generation, attempts "
            "FROM round_processing_jobs"
        ).fetchone() == ("processing", "worker-b", 2, 2)
        assert connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM core_commands").fetchone()[0] == 0


def test_processing_worker_request_stop_stops_created_loop(tmp_path: Path) -> None:
    database = tmp_path / "stop.db"
    _bootstrap(database)
    worker = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(),
        worker_id="worker-1",
    )
    loop = worker.create_loop(poll_interval_seconds=0.01)
    loop.start()

    worker.request_stop()

    assert loop.join(timeout=1) is True


def _candidate(title: str) -> HypothesisCandidate:
    return HypothesisCandidate(
        title=title,
        target="operations",
        statement=f"Use the {title.lower()} rule.",
        rationale="Keeps the rollout deterministic.",
    )


def _run_success(
    database: Path,
    drafts: tuple[HypothesisCandidate, ...],
) -> tuple[tuple[str, int], ...]:
    _bootstrap(database)
    _ingest(database)
    result = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(drafts=drafts),
        worker_id="worker-1",
    ).process_once()
    assert result is not None
    assert result.status == "completed"
    with sqlite3.connect(database) as connection:
        return tuple(
            connection.execute(
                """
                SELECT h.content_digest, h.hypothesis_index
                FROM core_commands AS c
                JOIN hypotheses AS h ON h.core_command_id = c.command_id
                ORDER BY h.content_digest
                """
            ).fetchall()
        )


def test_worker_hypothesis_idempotency_is_independent_of_response_order(
    tmp_path: Path,
) -> None:
    first = _run_success(
        tmp_path / "ab.db",
        (_candidate("Alpha"), _candidate("Beta")),
    )
    second = _run_success(
        tmp_path / "ba.db",
        (_candidate("Beta"), _candidate("Alpha")),
    )

    assert first == second
    assert len(first) == 2
    assert [row[1] for row in first] == [0, 1]


def test_worker_deduplicates_repeated_hypothesis_candidates(
    tmp_path: Path,
) -> None:
    rows = _run_success(
        tmp_path / "duplicates.db",
        (_candidate("Alpha"), _candidate("Alpha"), _candidate("Beta")),
    )

    assert len(rows) == 2
    assert len({row[0] for row in rows}) == 2
    assert len({row[1] for row in rows}) == 2
