from __future__ import annotations

import sqlite3
from pathlib import Path

from ledgermind_core.application.digests import calculate_raw_round_digest
from ledgermind_core.contracts import RawRoundRequest

from ledgermind_local.persistence import (
    SQLiteUnitOfWork,
    migrations,
    open_sqlite_connection,
)
from ledgermind_local.raw_rounds import RawRoundIngestHandler

_FIXTURE = {
    "api_version": "2",
    "idempotency_key": "sha256:" + "a" * 64,
    "memory_space_id": "retention-space",
    "source": {
        "system": "hermes",
        "instance_id": "instance",
        "profile_id": "profile",
        "session_id": "session",
        "round_id": "round",
        "first_event_id": "event-1",
        "final_event_id": "event-2",
        "event_ids": ["event-1", "event-2"],
        "source_schema_version": 1,
        "adapter_version": "test/1",
    },
    "round": {
        "started_at": "2026-08-02T20:00:00Z",
        "completed_at": "2026-08-02T20:01:00Z",
        "events": [
            {"event_id": "event-1", "sequence": 0, "kind": "message", "role": "user", "content": [{"type": "text", "text": "request"}]},
            {"event_id": "event-2", "sequence": 1, "kind": "message", "role": "assistant", "final": True, "content": [{"type": "text", "text": "response"}]},
        ],
    },
    "payload_digest": "sha256:" + "0" * 64,
}
_FIXTURE["payload_digest"] = calculate_raw_round_digest(_FIXTURE)


def _bootstrap(path: Path) -> None:
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def test_expired_raw_round_without_hypothesis_is_purged(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    request = RawRoundRequest.model_validate(_FIXTURE)
    result = RawRoundIngestHandler(database_path=database, retention_days=1).handle(request)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE raw_rounds SET retention_expires_at = ? WHERE raw_round_id = ?",
            ("2020-01-01T00:00:00+00:00", result.raw_round_id),
        )
        connection.commit()

    with SQLiteUnitOfWork(database, write_transaction=True) as uow:
        purged = uow.raw_rounds.purge_expired(now="2021-01-01T00:00:00+00:00")
        uow.commit()

    assert purged == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_rounds").fetchone()[0] == 0
