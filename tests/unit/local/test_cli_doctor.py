"""Tests for the `ledgermind doctor` command."""

from __future__ import annotations

from pathlib import Path

from ledgermind_local.cli import main
from ledgermind_local.persistence import (
    Atom,
    Knowledge,
    KnowledgeEvidence,
    KnowledgeRevision,
    SQLiteUnitOfWork,
    migrations,
)


def _make_atom(atom_id: str, memory_space_id: str) -> Atom:
    return Atom(
        atom_id=atom_id,
        memory_space_id=memory_space_id,
        source_system="hermes",
        source_instance_id="instance-1",
        source_profile_id="profile-1",
        source_session_id="session-1",
        source_round_id="round-1",
        source_round_key=f"round-key-{atom_id}",
        first_message_id=None,
        final_message_id="message-1",
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
        title="Atom",
        target="target-a",
        statement="Statement",
        rationale="Rationale",
        result="result",
        artifacts=("artifact-1",),
        content_digest="sha256:2222222222222222222222222222222222222222222222222222222222222222222222",
        supersedes_atom_id=None,
        created_at="2026-08-01T00:00:00Z",
    )


def _make_knowledge(knowledge_id: str, memory_space_id: str) -> Knowledge:
    return Knowledge(
        knowledge_id=knowledge_id,
        memory_space_id=memory_space_id,
        title="Knowledge",
        target="target-a",
        statement="Statement",
        rationale="Rationale",
        phase="pattern",
        version=1,
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
        superseded_by_id=None,
        deleted_at=None,
    )


def _make_revision(revision_id: str, knowledge_id: str) -> KnowledgeRevision:
    return KnowledgeRevision(
        revision_id=revision_id,
        knowledge_id=knowledge_id,
        version=1,
        event_type="KnowledgeCreated",
        snapshot_json=f'{{"knowledge_id":"{knowledge_id}"}}',
        cause_atom_id=None,
        created_at="2026-08-01T00:02:00Z",
    )


def _build_valid_db(path: Path) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        atom = _make_atom("atom-1", "space-a")
        knowledge = _make_knowledge("knw-1", "space-a")

        uow.atoms.add(atom)
        uow.knowledge.add(knowledge)
        uow.evidence.add(
            KnowledgeEvidence(
                knowledge_id=knowledge.knowledge_id,
                atom_id=atom.atom_id,
                relation="origin",
                created_at="2026-08-01T00:01:00Z",
            )
        )
        uow.revisions.add(_make_revision("rev-1", knowledge.knowledge_id))
        uow.commit()


def _build_orphan_db(path: Path) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(_make_atom("atom-1", "space-a"))
        uow.commit()


def test_doctor_returns_success_for_clean_database(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_valid_db(db_path)

    assert (
        main(["doctor", "--database", str(db_path)]) == 0
    )


def test_doctor_returns_error_on_orphan_atoms(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _build_orphan_db(db_path)

    assert main(["doctor", "--database", str(db_path)]) == 1
