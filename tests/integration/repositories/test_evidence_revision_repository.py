"""Integration tests for SQLite evidence and revision repositories."""

from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.persistence import (
    Atom,
    Knowledge,
    KnowledgeEvidence,
    KnowledgeRevision,
    SQLiteUnitOfWork,
    migrations,
)


def _knowledge_payload(
    *,
    knowledge_id: str = "knowledge-1",
    memory_space_id: str = "space-a",
) -> Knowledge:
    return Knowledge(
        knowledge_id=knowledge_id,
        memory_space_id=memory_space_id,
        title="First knowledge",
        target="target-a",
        statement="A implies B",
        rationale="observed in test",
        phase="pattern",
        version=1,
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
        superseded_by_id=None,
        deleted_at=None,
    )


def _atom_payload(
    *,
    atom_id: str = "atom-1",
    memory_space_id: str = "space-a",
    source_round_key: str = "round-key-1",
) -> Atom:

    return Atom(
        atom_id=atom_id,
        memory_space_id=memory_space_id,
        source_system="hermes",
        source_instance_id="instance-1",
        source_profile_id="profile-1",
        source_session_id="session-1",
        source_round_id="round-1",
        source_round_key=source_round_key,
        first_message_id=None,
        final_message_id="message-2",
        message_ids=("m1", "m2"),
        source_digest="sha256:1111111111111111111111111111111111111111111111111111111111111111111111",
        source_schema_version=1,
        resolver_version=1,
        extraction_host="local-test",
        extraction_provider="provider",
        extraction_model="model",
        extraction_prompt_version=1,
        extraction_schema_version=1,
        extraction_purpose="ledgermind.atom.extract",
        title="Пример",
        target="target-a",
        statement="A implies B",
        rationale="reason",
        result="result",
        artifacts=("artifact-1",),
        content_digest="sha256:2222222222222222222222222222222222222222222222222222222222222222222222",
        supersedes_atom_id=None,
        created_at="2026-08-01T00:00:00Z",
    )


def _evidence_payload(
    *,
    knowledge_id: str = "knowledge-1",
    atom_id: str = "atom-1",
    relation: str = "origin",
    created_at: str = "2026-08-01T00:01:00Z",
) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        knowledge_id=knowledge_id,
        atom_id=atom_id,
        relation=relation,
        created_at=created_at,
    )


def _revision_payload(
    *,
    revision_id: str = "rev-1",
    knowledge_id: str = "knowledge-1",
    version: int = 1,
    event_type: str = "KnowledgeCreated",
    snapshot_json: str = '{"memory_space_id":"space-a","title":"First knowledge"}',
    cause_atom_id: str | None = "atom-1",
    created_at: str = "2026-08-01T00:01:00Z",
) -> KnowledgeRevision:
    return KnowledgeRevision(
        revision_id=revision_id,
        knowledge_id=knowledge_id,
        version=version,
        event_type=event_type,
        snapshot_json=snapshot_json,
        cause_atom_id=cause_atom_id,
        created_at=created_at,
    )


def test_evidence_foreign_keys(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(_atom_payload())
        uow.knowledge.add(_knowledge_payload())
        uow.evidence.add(_evidence_payload())

        with pytest.raises(sqlite3.IntegrityError):
            uow.evidence.add(_evidence_payload(knowledge_id="missing-knowledge"))

        with pytest.raises(sqlite3.IntegrityError):
            uow.evidence.add(_evidence_payload(atom_id="missing-atom"))


def test_evidence_unique_relation(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(_atom_payload())
        uow.knowledge.add(_knowledge_payload())
        uow.evidence.add(_evidence_payload())

        with pytest.raises(sqlite3.IntegrityError):
            uow.evidence.add(_evidence_payload())


def test_revision_foreign_keys_and_snapshot(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.knowledge.add(_knowledge_payload())
        uow.atoms.add(_atom_payload())

        raw_snapshot = '{"b":2,   "a":1}'
        uow.revisions.add(_revision_payload(snapshot_json=raw_snapshot))

        with pytest.raises(sqlite3.IntegrityError):
            uow.revisions.add(_revision_payload(revision_id="rev-2", knowledge_id="missing"))

        revisions = uow.revisions.list_for_knowledge("space-a", "knowledge-1")
        assert len(revisions) == 1
        assert revisions[0].snapshot_json == raw_snapshot


def test_revision_unique_version(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.knowledge.add(_knowledge_payload())
        uow.atoms.add(_atom_payload())
        uow.revisions.add(_revision_payload())
        uow.revisions.add(_revision_payload(revision_id="rev-2", version=2))

        with pytest.raises(sqlite3.IntegrityError):
            uow.revisions.add(_revision_payload(revision_id="rev-3", version=1))
