"""Integration tests for core ingestion flow backed by local SQLite."""

from __future__ import annotations

import sqlite3

import pytest
from ledgermind_core.application.ingest_atom import IngestAtomResult
from ledgermind_core.application.mappers import IngestAtomCommand
from ledgermind_core.domain import AtomContent, ExtractionInfo, Phase, SourceReference
from ledgermind_core.domain.events import AtomCreated, KnowledgeCreated

from ledgermind_local.bootstrap import build_ingest_atom_handler
from ledgermind_local.persistence import SQLiteUnitOfWork, migrations

_MEMORY_SPACE_ID = "space-a"
_IDEMPOTENCY_KEY = "sha256:" + "a" * 64
_REQUEST_HASH = "sha256:" + "b" * 64


def _build_command(
    *,
    idempotency_key: str = _IDEMPOTENCY_KEY,
    request_hash: str = _REQUEST_HASH,
) -> IngestAtomCommand:
    return IngestAtomCommand(
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        memory_space_id=_MEMORY_SPACE_ID,
        source=SourceReference(
            source_system="hermes",
            source_instance_id="instance-1",
            source_profile_id="profile-1",
            source_session_id="session-1",
            source_round_id="round-1",
            first_message_id=None,
            final_message_id="message-1",
            message_ids=("m-1", "m-2"),
            source_digest="sha256:" + "c" * 64,
            source_schema_version=1,
            resolver_version=1,
        ),
        content=AtomContent(
            title="How to keep canonical memory",
            target="architecture.persistence",
            statement="Local SQLite should be the source of truth for stage 3.14.",
            rationale="Single source, durable events.",
            result="Initial implementation validated.",
            artifacts=("artifact-1", "artifact-2"),
        ),
        extraction=ExtractionInfo(
            host="hermes",
            provider="openrouter",
            model="gpt",
            prompt_version=1,
            schema_version=1,
            purpose="ledgermind.atom.extract",
        ),
    )


def _bootstrap_database(path) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.commit()


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()["total"])


def test_ingest_with_sqlite_is_persistent_and_idempotent(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    _bootstrap_database(db_path)
    command = _build_command()

    handler = build_ingest_atom_handler(database_path=db_path)
    first = handler.handle(command)
    handler = build_ingest_atom_handler(database_path=db_path)
    second = handler.handle(command)

    assert second == IngestAtomResult(
        atom_id=first.atom_id,
        knowledge_id=first.knowledge_id,
        knowledge_version=first.knowledge_version,
        phase=first.phase,
        duplicate=True,
        projections_pending=first.projections_pending,
    )

    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)

        atom = uow.atoms.get(first.atom_id, _MEMORY_SPACE_ID)
        knowledge = uow.knowledge.get(first.knowledge_id, _MEMORY_SPACE_ID)

        assert atom is not None
        assert knowledge is not None
        assert atom.title == command.content.title
        assert atom.target == command.content.target
        assert atom.statement == command.content.statement
        assert atom.rationale == command.content.rationale
        assert atom.result == command.content.result
        assert atom.extraction_host == command.extraction.host
        assert atom.extraction_provider == command.extraction.provider
        assert atom.extraction_model == command.extraction.model
        assert atom.extraction_prompt_version == command.extraction.prompt_version
        assert atom.extraction_schema_version == command.extraction.schema_version
        assert atom.extraction_purpose == command.extraction.purpose
        assert knowledge.phase == Phase.PATTERN.value
        assert uow.evidence.count_for_knowledge(_MEMORY_SPACE_ID, first.knowledge_id) == 1
        assert len(uow.revisions.list_for_knowledge(_MEMORY_SPACE_ID, first.knowledge_id)) == 1
        assert uow.idempotency.get(_MEMORY_SPACE_ID, _IDEMPOTENCY_KEY) is not None

        uow.commit()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        events = conn.execute(
            "SELECT event_type, aggregate_id FROM outbox_events ORDER BY event_type ASC"
        ).fetchall()
        assert len(events) == 2
        assert {row["event_type"] for row in events} == {
            AtomCreated.EVENT_NAME,
            KnowledgeCreated.EVENT_NAME,
        }
        assert {row["aggregate_id"] for row in events} == {first.atom_id, first.knowledge_id}
        deliveries = conn.execute(
            "SELECT COUNT(*) AS total FROM projection_deliveries"
        ).fetchone()["total"]
        assert deliveries >= 2
        assert _count_rows(conn, "atoms") == 1
        assert _count_rows(conn, "knowledge_items") == 1
        assert _count_rows(conn, "knowledge_evidence") == 1
        assert _count_rows(conn, "knowledge_revisions") == 1
        assert _count_rows(conn, "idempotency_results") == 1
    finally:
        conn.close()


def test_ingest_with_sqlite_rolls_back_when_outbox_write_fails(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "state.db"
    _bootstrap_database(db_path)
    command = _build_command()

    def _broken_outbox_add(self, event):
        raise RuntimeError("outbox write failed")

    monkeypatch.setattr("ledgermind_local.bootstrap._CoreOutboxEventRepository.add", _broken_outbox_add)

    handler = build_ingest_atom_handler(database_path=db_path)
    with pytest.raises(RuntimeError, match="outbox write failed"):
        handler.handle(command)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for table_name in (
            "atoms",
            "knowledge_items",
            "knowledge_evidence",
            "knowledge_revisions",
            "idempotency_results",
            "outbox_events",
            "projection_deliveries",
            "memory_spaces",
        ):
            assert _count_rows(conn, table_name) == 0
    finally:
        conn.close()
