from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ledgermind_protocol import RawRoundRequest, calculate_raw_round_digest

from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
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
        "idempotency_key": "sha256:" + "0" * 64,
        "memory_space_id": "lease-space",
        "source": {
            "system": "hermes",
            "instance_id": "instance",
            "profile_id": "profile",
            "session_id": "session",
            "round_id": "round",
            "first_event_id": "user-event",
            "final_event_id": "assistant-event",
            "event_ids": ["user-event", "assistant-event"],
            "source_schema_version": 1,
            "adapter_version": "test/1",
        },
        "round": {
            "started_at": "2026-08-03T00:00:00Z",
            "completed_at": "2026-08-03T00:01:00Z",
            "events": [
                {
                    "event_id": "user-event",
                    "sequence": 0,
                    "kind": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "request"}],
                },
                {
                    "event_id": "assistant-event",
                    "sequence": 1,
                    "kind": "message",
                    "role": "assistant",
                    "final": True,
                    "content": [{"type": "text", "text": "response"}],
                },
            ],
        },
        "payload_digest": "sha256:" + "0" * 64,
    }
    payload["payload_digest"] = calculate_raw_round_digest(payload)
    payload["idempotency_key"] = payload["payload_digest"]
    return RawRoundRequest.model_validate(payload)


def test_worker_renews_lease_during_generator_call(tmp_path: Path) -> None:
    database = tmp_path / "rounds.db"
    _bootstrap(database)
    result = RawRoundIngestHandler(database_path=database).handle(_request())
    started = threading.Event()
    release = threading.Event()

    class _SlowGenerator:
        provider = "test"
        model = "slow"
        prompt_version = 1
        schema_version = 1

        def generate(self, normalized):
            assert normalized.transcript == "user: request\nassistant: response"
            started.set()
            assert release.wait(timeout=4)
            return ()

    worker = RoundProcessingWorker(
        database_path=database,
        generator=_SlowGenerator(),
        worker_id="worker-1",
        lease_seconds=1,
        heartbeat_interval_seconds=0.1,
    )
    processing_result: list[object] = []
    thread = threading.Thread(
        target=lambda: processing_result.append(worker.process_once()),
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=2)
    time.sleep(1.2)

    with sqlite3.connect(database) as connection:
        lease_expires_at = connection.execute(
            "SELECT lease_expires_at FROM round_processing_jobs WHERE job_id = ?",
            (result.job_id,),
        ).fetchone()[0]
    assert datetime.fromisoformat(lease_expires_at) > datetime.now(timezone.utc)

    release.set()
    thread.join(timeout=4)
    assert not thread.is_alive()
    assert processing_result[0].status == "no_knowledge"
