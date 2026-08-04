from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ledgermind_local.core_gateway.model_task_contracts import SubmitModelResultCommand
from ledgermind_local.core_gateway.process import ProcessCoreGateway
from ledgermind_local.core_gateway.supervisor import CoreSupervisor
from ledgermind_local.inference.schemas import MergeProposal
from ledgermind_local.persistence import rounds_migrations
from ledgermind_local.scheduler.core_model_task_worker import CoreModelTaskWorker


class _Broker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_merge_proposal(self, **kwargs: object) -> MergeProposal:
        self.calls.append(kwargs)
        return MergeProposal(
            title="Merged operational knowledge",
            target="ops",
            statement="Both source items describe the same operational pattern.",
            rationale="The worker preserves both source references and the required constraint.",
            preserved_references=("knowledge-a", "knowledge-b"),
            preserved_constraints=("keep source",),
        )


def _daemon_binary() -> Path:
    configured = os.environ.get("LEDGERMIND_CORE_DAEMON")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[3]
        / "ledgermind-core"
        / "target"
        / "debug"
        / "ledgermind-core"
    )


def _seed_local_rounds(database: Path, now: str) -> None:
    connection = sqlite3.connect(database)
    rounds_migrations.apply_migrations(connection)
    connection.execute(
        """
        INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        ("space-runtime", "integration-test", now, now),
    )
    connection.execute(
        """
        INSERT INTO memory_space_inference_profiles(
            memory_space_id, hypothesis_profile_id, merge_profile_id, updated_at
        ) VALUES (?, ?, ?, ?)
        """,
        ("space-runtime", None, "merge-default", now),
    )
    connection.commit()
    connection.close()


def _seed_core_database(database: Path, now: datetime, expires_at: str) -> None:
    connection = sqlite3.connect(database)
    timestamp = now.isoformat().replace("+00:00", "Z")
    connection.execute(
        """
        INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        ("space-runtime", "integration-test", timestamp, timestamp),
    )
    for knowledge_id, title, version in (
        ("knowledge-a", "Knowledge A", 1),
        ("knowledge-b", "Knowledge B", 2),
    ):
        connection.execute(
            """
            INSERT INTO knowledge_items(
                knowledge_id, memory_space_id, title, target, statement, rationale,
                phase, version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                knowledge_id,
                "space-runtime",
                title,
                "ops",
                f"Statement for {title}",
                "Source rationale",
                "pattern",
                version,
                timestamp,
                timestamp,
            ),
        )
    task_id = "task-runtime-1"
    payload = {
        "task_id": task_id,
        "operation": "merge_knowledge",
        "memory_space_id": "space-runtime",
        "expected_versions": {"knowledge-a": 1, "knowledge-b": 2},
        "expires_at": expires_at,
        "model_input": {
            "items": [
                {"reference": "knowledge-a", "required_constraints": ["keep source"]},
                {"reference": "knowledge-b", "required_constraints": ["keep source"]},
            ]
        },
    }
    connection.execute(
        """
        INSERT INTO model_tasks(
            task_id, memory_space_id, task_type, status, request_digest,
            payload_json, created_at, updated_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            "space-runtime",
            "merge_knowledge",
            "queued",
            "sha256:" + ("a" * 64),
            json.dumps(payload, separators=(",", ":")),
            timestamp,
            timestamp,
            expires_at,
        ),
    )
    connection.commit()
    connection.close()


@pytest.mark.integration
def test_real_daemon_local_worker_accepts_merge_task(tmp_path: Path) -> None:
    daemon = _daemon_binary()
    if not daemon.is_file():
        pytest.skip("build ledgermind-core or set LEDGERMIND_CORE_DAEMON")

    core_database = tmp_path / "knowledge.db"
    rounds_database = tmp_path / "rounds.db"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    timestamp = now.isoformat().replace("+00:00", "Z")
    expires_at = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    supervisor = CoreSupervisor(
        [str(daemon), "--database", str(core_database)],
        core_data_dir=tmp_path,
    )
    gateway = ProcessCoreGateway(supervisor)
    broker = _Broker()
    try:
        supervisor.start()
        _seed_core_database(core_database, now, expires_at)
        _seed_local_rounds(rounds_database, timestamp)
        worker = CoreModelTaskWorker(
            database_path=rounds_database,
            gateway=gateway,
            broker=broker,
            worker_id="local-model-tasks",
        )

        stats = worker.process_once()

        assert stats.fetched == 1
        assert stats.completed == 1
        assert stats.duplicates == 0
        assert stats.failed == 0
        assert broker.calls[0]["profile_id"] == "merge-default"
        assert broker.calls[0]["memory_space_id"] == "space-runtime"

        connection = sqlite3.connect(core_database)
        row = connection.execute(
            "SELECT status, result_json, lease_owner, lease_expires_at FROM model_tasks WHERE task_id = ?",
            ("task-runtime-1",),
        ).fetchone()
        connection.close()
        assert row is not None
        assert row[0] == "completed"
        assert json.loads(row[1])["preserved_references"] == [
            "knowledge-a",
            "knowledge-b",
        ]
        assert row[2] is None
        assert row[3] is None

        connection = sqlite3.connect(core_database)
        successor = connection.execute(
            "SELECT knowledge_id, phase, version, superseded_by_id FROM knowledge_items "
            "WHERE title = ?",
            ("Merged operational knowledge",),
        ).fetchone()
        assert successor is not None
        successor_id, phase, version, successor_link = successor
        assert phase == "emergent"
        assert version == 1
        assert successor_link is None
        for knowledge_id in ("knowledge-a", "knowledge-b"):
            source = connection.execute(
                "SELECT version, superseded_by_id FROM knowledge_items WHERE knowledge_id = ?",
                (knowledge_id,),
            ).fetchone()
            assert source == (3 if knowledge_id == "knowledge-b" else 2, successor_id)
        assert connection.execute("SELECT COUNT(*) FROM knowledge_revisions").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM supersession_links").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM projection_events").fetchone()[0] == 3
        connection.close()

        duplicate = gateway.submit_model_result(
            SubmitModelResultCommand(
                request_id="runtime-duplicate",
                task_id="task-runtime-1",
                memory_space_id="space-runtime",
                worker_id="local-model-tasks",
                result={
                    "title": "Merged operational knowledge",
                    "target": "ops",
                    "statement": "Both source items describe the same operational pattern.",
                    "rationale": "The worker preserves both source references and the required constraint.",
                    "preserved_references": ["knowledge-a", "knowledge-b"],
                    "preserved_constraints": ["keep source"],
                },
            )
        )
        assert duplicate.accepted is True
        assert duplicate.duplicate is True
        connection = sqlite3.connect(core_database)
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_items WHERE title = ?",
            ("Merged operational knowledge",),
        ).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM supersession_links").fetchone()[0] == 2
        connection.close()
    finally:
        gateway.close()
