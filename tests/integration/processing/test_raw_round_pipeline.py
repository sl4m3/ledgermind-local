from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient
from ledgermind_core.application.digests import calculate_raw_round_digest

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Application, Settings
from ledgermind_local.bootstrap import build_round_processing_worker
from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.processing.generator import HypothesisDraft, HypothesisGenerator
from ledgermind_local.processing.models import NormalizedRound

_ROUND = {
    "api_version": "2",
    "idempotency_key": "sha256:" + "a" * 64,
    "memory_space_id": "workspace-e2e",
    "source": {
        "system": "hermes",
        "instance_id": "src-hermes-local",
        "profile_id": "default",
        "session_id": "session-e2e",
        "round_id": "round-e2e",
        "first_event_id": "event-1",
        "final_event_id": "event-2",
        "event_ids": ["event-1", "event-2"],
        "source_schema_version": 1,
        "adapter_version": "hermes-test/1",
    },
    "round": {
        "started_at": "2026-08-02T20:00:00Z",
        "completed_at": "2026-08-02T20:01:00Z",
        "events": [
            {
                "event_id": "event-1",
                "sequence": 0,
                "kind": "message",
                "role": "user",
                "content": [{"type": "text", "text": "capture request"}],
            },
            {
                "event_id": "event-2",
                "sequence": 1,
                "kind": "message",
                "role": "assistant",
                "final": True,
                "content": [{"type": "text", "text": "capture response"}],
            },
        ],
    },
    "payload_digest": "sha256:" + "0" * 64,
}


class _Generator:
    provider = "e2e-provider"
    model = "e2e-model"
    prompt_version = 2
    schema_version = 3

    def generate(self, normalized: NormalizedRound) -> tuple[HypothesisDraft, ...]:
        assert normalized.transcript.startswith("user: capture request")
        return (
            HypothesisDraft(
                title="Deployment rule",
                target="operations",
                statement="Deployments require explicit staging validation.",
                rationale="The captured round contains an operational deployment request.",
                result="Use staging validation before production.",
                artifacts=("runbook",),
            ),
        )


def _bootstrap(path: Path) -> None:
    with open_sqlite_connection(path) as connection:
        migrations.apply_migrations(connection)
        connection.commit()


def test_post_round_processing_bridge_persists_knowledge(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap(database)
    payload = dict(_ROUND)
    payload["payload_digest"] = calculate_raw_round_digest(payload)
    payload["idempotency_key"] = payload["payload_digest"]

    client = TestClient(
        create_app(
            application=cast(Application, object()),
            settings=Settings(database_path=database, api_token="token"),
        )
    )
    response = client.post(
        "/v1/rounds",
        headers={"Authorization": "Bearer token"},
        json=payload,
    )
    assert response.status_code == 202

    result = build_round_processing_worker(
        database_path=database,
        generator=cast(HypothesisGenerator, _Generator()),
        retry_delay_seconds=0,
    ).process_once()

    assert result is not None
    assert result.status == "completed"
    assert len(result.hypothesis_ids) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_rounds").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM round_processing_jobs WHERE status = 'completed'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0] == 1
        assert connection.execute(
            "SELECT source_client FROM memory_spaces WHERE memory_space_id = ?",
            ("workspace-e2e",),
        ).fetchone()[0] == "ledgermind-integrations"
