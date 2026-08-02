"""Integration tests for SQLite integrity diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ledgermind_local.diagnostics.integrity import run_database_integrity_checks
from ledgermind_local.persistence import (
    Atom,
    Knowledge,
    KnowledgeEvidence,
    KnowledgeRevision,
    SQLiteUnitOfWork,
    migrations,
    open_sqlite_connection,
)


def _build_atom(*, atom_id: str, memory_space_id: str) -> Atom:
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
        title="Atom title",
        target="target-a",
        statement="A implies B",
        rationale="rationale",
        result="result",
        artifacts=("artifact-1",),
        content_digest="sha256:2222222222222222222222222222222222222222222222222222222222222222222222",
        supersedes_atom_id=None,
        created_at="2026-08-01T00:00:00Z",
    )


def _build_knowledge(
    *,
    knowledge_id: str,
    memory_space_id: str,
    version: int = 1,
    superseded_by_id: str | None = None,
) -> Knowledge:
    return Knowledge(
        knowledge_id=knowledge_id,
        memory_space_id=memory_space_id,
        title="First knowledge",
        target="target-a",
        statement="Statement",
        rationale="Rationale",
        phase="pattern",
        version=version,
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
        superseded_by_id=superseded_by_id,
        deleted_at=None,
    )


def _build_evidence(*, knowledge_id: str, atom_id: str) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        knowledge_id=knowledge_id,
        atom_id=atom_id,
        relation="origin",
        created_at="2026-08-01T00:01:00Z",
    )


def _build_revision(*, revision_id: str, knowledge_id: str, version: int) -> KnowledgeRevision:
    return KnowledgeRevision(
        revision_id=revision_id,
        knowledge_id=knowledge_id,
        version=version,
        event_type="KnowledgeCreated",
        snapshot_json=f'{{"knowledge_id":"{knowledge_id}"}}',
        cause_atom_id=None,
        created_at="2026-08-01T00:02:00Z",
    )


def _run_integrity_checks(
    path: Path,
    **kwargs: Any,
) -> list[str]:
    connection = open_sqlite_connection(path)
    try:
        return run_database_integrity_checks(connection, **kwargs)
    finally:
        connection.close()


def _bootstrap_database(path: Path) -> None:
    with SQLiteUnitOfWork(path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.commit()


def test_integrity_passes_for_clean_database(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        atom = _build_atom(atom_id="atom-1", memory_space_id="space-a")
        knowledge = _build_knowledge(knowledge_id="knw-1", memory_space_id="space-a")
        evidence = _build_evidence(knowledge_id=knowledge.knowledge_id, atom_id=atom.atom_id)
        revision = _build_revision(revision_id="rev-1", knowledge_id=knowledge.knowledge_id, version=1)

        uow.atoms.add(atom)
        uow.knowledge.add(knowledge)
        uow.evidence.add(evidence)
        uow.revisions.add(revision)
        uow.commit()

    issues = _run_integrity_checks(db_path)
    assert issues == []


def test_orphan_atoms_are_reported_unless_reason_is_documented(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _bootstrap_database(db_path)
    with SQLiteUnitOfWork(db_path) as uow:
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(_build_atom(atom_id="atom-1", memory_space_id="space-a"))
        uow.commit()

    errors_without_reason = _run_integrity_checks(db_path)
    assert any("orphan atoms detected" in issue for issue in errors_without_reason)

    errors_with_reason = _run_integrity_checks(db_path, allow_orphan_atoms_reason="migration")
    assert not errors_with_reason


def test_current_knowledge_must_have_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        uow.knowledge.add(
            _build_knowledge(knowledge_id="knw-no-evidence", memory_space_id="space-a")
        )
        uow.revisions.add(
            _build_revision(
                revision_id="rev-no-evidence",
                knowledge_id="knw-no-evidence",
                version=1,
            )
        )
        uow.commit()

    issues = _run_integrity_checks(db_path)
    assert any("current knowledge missing evidence" in issue for issue in issues)


def test_knowledge_version_must_match_revision_max(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        knowledge = _build_knowledge(
            knowledge_id="knw-version",
            version=2,
            memory_space_id="space-a",
        )
        uow.knowledge.add(knowledge)
        atom = _build_atom(atom_id="atom-1", memory_space_id="space-a")
        uow.atoms.add(atom)
        uow.revisions.add(
            _build_revision(
                revision_id="rev-1",
                knowledge_id=knowledge.knowledge_id,
                version=1,
            )
        )
        uow.evidence.add(
            _build_evidence(knowledge_id=knowledge.knowledge_id, atom_id="atom-1")
        )
        uow.commit()

    issues = _run_integrity_checks(db_path)
    assert any("knowledge version mismatch" in issue for issue in issues)


def test_superseded_by_cycles_are_reported(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        for knowledge_id in ("knw-1", "knw-2", "knw-3"):
            uow.knowledge.add(
                _build_knowledge(
                    knowledge_id=knowledge_id,
                    memory_space_id="space-a",
                    superseded_by_id=None,
                )
            )
            uow.revisions.add(
                _build_revision(
                    revision_id=f"rev-{knowledge_id}",
                    knowledge_id=knowledge_id,
                    version=1,
                )
            )

        uow.connection.execute(
            """
            UPDATE knowledge_items
            SET superseded_by_id = CASE knowledge_id
                WHEN 'knw-1' THEN 'knw-2'
                WHEN 'knw-2' THEN 'knw-3'
                WHEN 'knw-3' THEN 'knw-1'
            END
            WHERE knowledge_id IN ('knw-1', 'knw-2', 'knw-3')
            """
        )
        atom = _build_atom(atom_id="atom-1", memory_space_id="space-a")
        uow.atoms.add(atom)
        uow.connection.execute(
            """
            INSERT INTO knowledge_evidence (knowledge_id, atom_id, relation, created_at)
            VALUES ('knw-1','atom-1','origin','2026-08-01T00:01:00Z')
            """
        )
        uow.commit()

    issues = _run_integrity_checks(db_path)
    assert any("superseded_by_id cycle" in issue for issue in issues)


def test_evidence_relations_must_match_memory_space(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.memory_spaces.ensure("space-b", "hermes")

        atom = _build_atom(atom_id="atom-b", memory_space_id="space-b")
        knowledge = _build_knowledge(knowledge_id="knw-b", memory_space_id="space-a")

        uow.atoms.add(atom)
        uow.knowledge.add(knowledge)
        uow.evidence.add(_build_evidence(knowledge_id=knowledge.knowledge_id, atom_id=atom.atom_id))
        uow.revisions.add(
            _build_revision(
                revision_id="rev-b",
                knowledge_id=knowledge.knowledge_id,
                version=1,
            )
        )
        uow.commit()

    issues = _run_integrity_checks(db_path)
    assert any("evidence crosses memory spaces" in issue for issue in issues)
