from __future__ import annotations

import os
from pathlib import Path

import pytest

from ledgermind_local.core_gateway.contracts import (
    AcceptHypothesisCommand,
    HypothesisEvidence,
    HypothesisExtraction,
    HypothesisPayload,
    RecordContextUsageCommand,
    RetrieveContextCommand,
)
from ledgermind_local.core_gateway.process import ProcessCoreGateway
from ledgermind_local.core_gateway.projection_contracts import (
    PollProjectionEventsCommand,
)
from ledgermind_local.core_gateway.supervisor import CoreSupervisor
from ledgermind_local.persistence import open_sqlite_connection, rounds_migrations
from ledgermind_local.projections import KnowledgeFTSProjection
from ledgermind_local.scheduler import CoreProjectionWorker
from ledgermind_local.search import CoreBackedSearch, CoreProjectionSearchAdapter


def _rust_daemon_path() -> Path:
    configured = os.environ.get("LEDGERMIND_RUST_DAEMON")
    if configured:
        return Path(configured)
    workspace_root = Path(__file__).resolve().parents[4]
    return workspace_root / "ledgermind-core" / "target" / "debug" / "ledgermind-core"


_RUST_DAEMON = _rust_daemon_path()
pytestmark = pytest.mark.skipif(
    not _RUST_DAEMON.is_file(),
    reason="Rust daemon binary is not built; set LEDGERMIND_RUST_DAEMON to enable",
)


def _accept_command() -> AcceptHypothesisCommand:
    digest = "sha256:" + "a" * 64
    return AcceptHypothesisCommand(
        protocol_version=1,
        command_id="local-rust-command-1",
        idempotency_key=digest,
        memory_space_id="local-rust-space",
        hypothesis=HypothesisPayload(
            hypothesis_id="local-rust-hypothesis-1",
            content_digest="sha256:" + "b" * 64,
            title="Local Rust context",
            target="Rust Core",
            statement="Local ProcessCoreGateway reaches Rust Core",
            rationale="Stage 10 integration",
            result="accepted",
            artifacts=(),
            evidence=HypothesisEvidence(
                source_system="local",
                source_instance_id="instance",
                source_profile_id="profile",
                source_session_id="session",
                source_round_id="round",
                raw_round_digest="sha256:" + "c" * 64,
                normalized_round_digest="sha256:" + "d" * 64,
                source_event_ids=("local-rust-event-1",),
            ),
            extraction=HypothesisExtraction(
                provider="local",
                model="test",
                prompt_version=1,
                schema_version=1,
                completed_at="2026-08-03T00:00:00Z",
            ),
        ),
    )


def test_process_gateway_uses_rust_daemon_vertical_slice(tmp_path: Path) -> None:
    supervisor = CoreSupervisor(
        [str(_RUST_DAEMON), "--database", str(tmp_path / "knowledge.db")],
        startup_timeout_seconds=2.0,
        operation_timeout_seconds=2.0,
        core_data_dir=tmp_path,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
        accepted = gateway.accept_hypothesis(_accept_command())
        context = gateway.retrieve_context(
            RetrieveContextCommand(
                request_id="local-rust-retrieve-1",
                memory_space_id="local-rust-space",
                query="Local Rust context",
                limit=5,
            )
        )
        gateway.record_context_usage(
            RecordContextUsageCommand(
                request_id="local-rust-usage-1",
                memory_space_id="local-rust-space",
                item_ids=(accepted.core_reference_id or "",),
            )
        )
    finally:
        gateway.close()

    assert health.healthy is True
    assert accepted.accepted is True
    assert accepted.duplicate is False
    assert accepted.core_reference_id
    assert len(context.items) == 1
    assert context.items[0].knowledge_id == accepted.core_reference_id


def test_rust_projection_event_replays_then_flows_through_local_worker(
    tmp_path: Path,
) -> None:
    core_database = tmp_path / "knowledge.db"
    rounds_database = tmp_path / "rounds.db"
    supervisor = CoreSupervisor(
        [str(_RUST_DAEMON), "--database", str(core_database)],
        startup_timeout_seconds=2.0,
        operation_timeout_seconds=2.0,
        core_data_dir=tmp_path,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        accepted = gateway.accept_hypothesis(_accept_command())
        first_poll = gateway.poll_projection_events(
            PollProjectionEventsCommand(
                request_id="projection-poll-1",
                memory_space_id="local-rust-space",
                consumer_id="local-projections:projections.search",
                limit=10,
            )
        )
        replay_poll = gateway.poll_projection_events(
            PollProjectionEventsCommand(
                request_id="projection-poll-2",
                memory_space_id="local-rust-space",
                consumer_id="local-projections:projections.search",
                limit=10,
            )
        )

        assert accepted.core_reference_id
        assert len(first_poll.events) == 1
        assert len(replay_poll.events) == 1
        parsed = first_poll.events[0].parse_payload()
        assert parsed.knowledge_id == accepted.core_reference_id
        assert parsed.memory_space_id == "local-rust-space"

        with open_sqlite_connection(rounds_database) as connection:
            rounds_migrations.apply_migrations(connection)
            connection.execute(
                """
                INSERT INTO memory_spaces (
                    memory_space_id, display_name, source_client, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "local-rust-space",
                    "Local Rust",
                    "integration-test",
                    "2026-08-03T00:00:00Z",
                    "2026-08-03T00:00:00Z",
                ),
            )
            connection.commit()

        worker = CoreProjectionWorker(
            database_path=rounds_database,
            gateway=gateway,
            consumer_id="local-projections",
            handlers_factory=lambda connection: {
                "projections.search": KnowledgeFTSProjection(connection)
            },
        )
        stats = worker.process_once()
        worker.close()

        assert stats.fetched == 1
        assert stats.acknowledged == 1
        assert stats.processed == 1
        with open_sqlite_connection(rounds_database) as connection:
            row = connection.execute(
                """
                SELECT knowledge_id, memory_space_id, title, target, statement
                FROM core_knowledge_fts
                WHERE knowledge_id = ? AND memory_space_id = ?
                """,
                (accepted.core_reference_id, "local-rust-space"),
            ).fetchone()
            assert row is not None
            assert next(iter(row)) == accepted.core_reference_id
            recorded_commands = []

            class RecordingGateway:
                def retrieve_context(self, command):
                    recorded_commands.append(command)
                    return gateway.retrieve_context(command)

            context = CoreBackedSearch(
                CoreProjectionSearchAdapter(connection),  # type: ignore[arg-type]
                RecordingGateway(),  # type: ignore[arg-type]
            ).retrieve_context(
                request_id="local-search-1",
                memory_space_id="local-rust-space",
                query="Local Rust context",
                limit=5,
            )
            assert len(context.items) == 1
            assert context.items[0].knowledge_id == accepted.core_reference_id
            assert len(recorded_commands) == 1
            assert recorded_commands[0].candidate_ids == (accepted.core_reference_id,)
            assert recorded_commands[0].candidate_scores == (
                (accepted.core_reference_id, 1.0),
            )

        after_ack = gateway.poll_projection_events(
            PollProjectionEventsCommand(
                request_id="projection-poll-3",
                memory_space_id="local-rust-space",
                consumer_id="local-projections:projections.search",
                limit=10,
            )
        )
        assert after_ack.events == ()
    finally:
        gateway.close()
