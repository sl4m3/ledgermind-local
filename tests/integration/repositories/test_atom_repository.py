"""Integration tests for SQLite atom repository."""

from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.persistence import (
    Atom,
    SQLiteAtomRepository,
    SQLiteUnitOfWork,
    migrations,
    open_sqlite_connection,
)


def _build_connection(path) -> sqlite3.Connection:
    connection = open_sqlite_connection(path)
    migrations.apply_migrations(connection)
    return connection


def _atom_payload(
    *,
    atom_id: str = "atom-1",
    memory_space_id: str = "space-a",
    source_round_key: str = "round-key-1",
    message_ids: tuple[str, ...] = ("m2", "m1"),
    supersedes_atom_id: str | None = None,
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
        message_ids=message_ids,
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
        supersedes_atom_id=supersedes_atom_id,
        created_at="2026-08-01T00:00:00Z",
    )


def test_atom_write_read_cycle(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        atom = _atom_payload(message_ids=("m1", "m2"))
        uow.memory_spaces.ensure(atom.memory_space_id, "hermes")
        uow.atoms.add(atom)
        assert uow.atoms.get("atom-1", atom.memory_space_id) == atom

        uow.commit()


def test_message_ids_stored_as_deterministic_json(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(
            _atom_payload(
                atom_id="atom-1",
                memory_space_id="space-a",
                source_round_key="round-key-1",
                message_ids=("z", "a", "a", "b"),
            )
        )
        uow.commit()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # check that duplicates are removed and order is deterministic
        raw = conn.execute(
            "SELECT message_ids_json FROM atoms WHERE atom_id = 'atom-1'"
        ).fetchone()[0]
        assert raw == '["a","b","z"]'
    finally:
        conn.close()


def test_extraction_version_uniqueness(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.atoms.add(_atom_payload())
        duplicate = _atom_payload(atom_id="atom-2")
        with pytest.raises(sqlite3.IntegrityError):
            uow.atoms.add(duplicate)

        uow.rollback()


def test_find_by_source_round(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        atom = _atom_payload(message_ids=("m1", "m2"))
        uow.memory_spaces.ensure(atom.memory_space_id, "hermes")
        uow.atoms.add(atom)
        found = uow.atoms.get_by_source_round(
            atom.memory_space_id,
            atom.source_round_key,
            extraction_prompt_version=1,
            extraction_schema_version=1,
        )
        assert found == atom
        uow.commit()


def test_atom_not_visible_across_memory_spaces(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.memory_spaces.ensure("space-b", "hermes")
        uow.atoms.add(_atom_payload(memory_space_id="space-a"))
        uow.atoms.add(
            _atom_payload(
                atom_id="atom-b",
                memory_space_id="space-b",
                source_round_key="round-key-2",
            )
        )
        assert uow.atoms.get("atom-1", "space-a") is not None
        assert uow.atoms.get("atom-1", "space-b") is None
        uow.commit()


def test_self_supersede_is_rejected_by_db(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        atom = _atom_payload(
            atom_id="self-atom",
            supersedes_atom_id="self-atom",
            source_round_key="self-replace-round",
        )
        with pytest.raises(sqlite3.IntegrityError):
            uow.atoms.add(atom)


def test_atom_repository_has_no_update_method(tmp_path) -> None:
    conn = _build_connection(tmp_path / "state.db")
    try:
        repo = SQLiteAtomRepository(conn)
        assert not hasattr(repo, "update")
    finally:
        conn.close()
