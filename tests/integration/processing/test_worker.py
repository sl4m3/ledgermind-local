from __future__ import annotations

import sqlite3
from pathlib import Path

from ledgermind_protocol import RawRoundRequest, calculate_raw_round_digest

from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.processing.generator import HypothesisDraft
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
                {"event_id": "m-1", "sequence": 0, "kind": "message", "role": "user", "content": [{"type": "text", "text": "request"}]},
                {"event_id": "m-2", "sequence": 1, "kind": "message", "role": "assistant", "final": True, "content": [{"type": "text", "text": "answer"}]},
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

    def __init__(self, drafts=(), error: Exception | None = None, probe=None) -> None:
        self.drafts = tuple(drafts)
        self.error = error
        self.probe = probe

    def generate(self, normalized):
        if self.probe is not None:
            self.probe()
        if self.error is not None:
            raise self.error
        return self.drafts


def _ingest(path: Path) -> None:
    result = RawRoundIngestHandler(database_path=path).handle(_request())
    assert result.duplicate is False


def test_worker_persists_hypothesis_after_callback_outside_write_lock(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    _ingest(database)
    lock_probe = {"ok": False}
    bridge_calls: list[str] = []

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
                HypothesisDraft(
                    title="Deployment rule",
                    target="operations",
                    statement="Deployments require a staging check token=secret.",
                    artifacts=("runbook",),
                ),
            ),
            probe=probe_lock,
        ),
        bridge=lambda raw, draft: bridge_calls.append(draft.title),
        worker_id="worker-1",
    )

    result = worker.process_once()

    assert result is not None
    assert result.status == "completed"
    assert lock_probe["ok"] is True
    assert bridge_calls == ["Deployment rule"]
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT status FROM round_processing_jobs").fetchone()
        assert row == ("completed",)
        hypothesis = connection.execute(
            "SELECT title, statement, artifacts_json FROM hypotheses"
        ).fetchone()
        assert hypothesis[0] == "Deployment rule"
        assert "secret" not in hypothesis[1]
        assert "[REDACTED]" in hypothesis[1]
        assert connection.execute("SELECT COUNT(*) FROM raw_rounds").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0] == 1


def test_worker_marks_no_knowledge_without_bridge(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    _ingest(database)
    bridge_calls: list[str] = []

    result = RoundProcessingWorker(
        database_path=database,
        generator=_Generator(),
        bridge=lambda raw, draft: bridge_calls.append(draft.title),
    ).process_once()

    assert result is not None
    assert result.status == "no_knowledge"
    assert bridge_calls == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM hypothesis_attempts").fetchone()[0] == 1


def test_worker_retries_generation_error_without_duplicate_raw_round(tmp_path: Path) -> None:
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
        assert connection.execute("SELECT attempts, status FROM round_processing_jobs").fetchone() == (1, "retry_wait")
        error_detail = connection.execute("SELECT error_detail FROM hypothesis_attempts").fetchone()[0]
        assert "secret" not in error_detail
        assert "[REDACTED]" in error_detail
