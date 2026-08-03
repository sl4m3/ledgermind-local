from __future__ import annotations

from types import SimpleNamespace

from ledgermind_local.persistence.raw_round_repository import RawRoundRecord
from ledgermind_local.processing.bridge import CoreHypothesisBridge
from ledgermind_local.processing.generator import HypothesisDraft


def test_core_bridge_maps_hypothesis_to_existing_ingest_command() -> None:
    captured = []

    class Handler:
        def handle(self, command):
            captured.append(command)
            return SimpleNamespace(atom_id="atom-1")

    raw = RawRoundRecord(
        raw_round_id="raw-1",
        memory_space_id="space",
        source_system="hermes",
        source_instance_id="instance",
        source_profile_id="profile",
        source_session_id="session",
        source_round_id="round",
        source_round_key="source-key",
        capture_schema_version=1,
        adapter_version="hermes/1",
        payload_json="{}",
        payload_digest="sha256:" + "a" * 64,
        started_at="2026-08-02T20:00:00+00:00",
        completed_at="2026-08-02T20:01:00+00:00",
        received_at="2026-08-02T20:01:01+00:00",
        retention_expires_at=None,
    )
    draft = HypothesisDraft(
        title="Title",
        target="Target",
        statement="Statement",
        rationale="Rationale",
        result="Result",
        artifacts=("artifact",),
    )

    result = CoreHypothesisBridge(Handler()).publish(raw, draft)

    assert result.atom_id == "atom-1"
    command = captured[0]
    assert command.memory_space_id == "space"
    assert command.source.source_system == "hermes"
    assert command.source.source_round_id.startswith("round:hypothesis:")
    assert command.source.source_digest == raw.payload_digest
    assert command.content.statement == "Statement"
    assert command.extraction.purpose == "ledgermind.hypothesis.bridge"
    assert command.idempotency_key.startswith("sha256:")
