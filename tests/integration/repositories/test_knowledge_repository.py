"""Integration tests for SQLite knowledge repository."""

from __future__ import annotations

import pytest

from ledgermind_local.persistence import (
    Knowledge,
    SQLiteKnowledgeConcurrencyError,
    SQLiteUnitOfWork,
    migrations,
)


def _knowledge_payload(
    *,
    knowledge_id: str = "knowledge-1",
    memory_space_id: str = "space-a",
    title: str = "First knowledge",
    target: str = "target-a",
    statement: str = "Statement A",
    rationale: str = "Rationale A",
    phase: str = "pattern",
    version: int = 1,
    created_at: str = "2026-08-01T00:00:00Z",
    updated_at: str = "2026-08-01T00:00:00Z",
    superseded_by_id: str | None = None,
    deleted_at: str | None = None,
) -> Knowledge:
    return Knowledge(
        knowledge_id=knowledge_id,
        memory_space_id=memory_space_id,
        title=title,
        target=target,
        statement=statement,
        rationale=rationale,
        phase=phase,
        version=version,
        created_at=created_at,
        updated_at=updated_at,
        superseded_by_id=superseded_by_id,
        deleted_at=deleted_at,
    )


def test_knowledge_write_read_cycle(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        expected = _knowledge_payload()
        uow.knowledge.add(expected)
        actual = uow.knowledge.get(expected.knowledge_id, expected.memory_space_id)

        assert actual == expected


def test_filter_by_memory_space(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        uow.memory_spaces.ensure("space-b", "hermes")

        uow.knowledge.add(_knowledge_payload(knowledge_id="space-a-1", memory_space_id="space-a"))
        uow.knowledge.add(_knowledge_payload(knowledge_id="space-a-2", memory_space_id="space-a"))
        uow.knowledge.add(_knowledge_payload(knowledge_id="space-b-1", memory_space_id="space-b"))

        space_a = uow.knowledge.list_by_space("space-a")
        space_b = uow.knowledge.list_by_space("space-b")
        assert {item.knowledge_id for item in space_a} == {"space-a-1", "space-a-2"}
        assert {item.knowledge_id for item in space_b} == {"space-b-1"}


def test_knowledge_optimistic_version(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        uow.knowledge.add(_knowledge_payload())
        updated = uow.knowledge.update(
            _knowledge_payload(
                title="Updated title",
                updated_at="2026-08-01T00:01:00Z",
            ),
            expected_version=1,
        )

        assert updated.version == 2

        with pytest.raises(SQLiteKnowledgeConcurrencyError):
            uow.knowledge.update(
                _knowledge_payload(
                    title="Concurrent title",
                    updated_at="2026-08-01T00:02:00Z",
                ),
                expected_version=1,
            )


def test_list_only_current_excludes_superseded_and_deleted(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        uow.knowledge.add(_knowledge_payload(knowledge_id="k1"))
        uow.knowledge.add(_knowledge_payload(knowledge_id="k2", title="Second", target="target-b"))
        uow.knowledge.add(_knowledge_payload(knowledge_id="k3", title="Third", target="target-c"))

        uow.knowledge.mark_superseded("k1", "space-a", superseded_by_id="k2")
        uow.knowledge.mark_deleted("k2", "space-a", deleted_at="2026-08-01T00:10:00Z")

        current = uow.knowledge.list_current_by_space("space-a")
        assert len(current) == 1
        assert current[0].knowledge_id == "k3"


def test_update_does_not_overwrite_created_at(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")
        original = _knowledge_payload(created_at="2026-08-01T00:00:00Z")
        uow.knowledge.add(original)

        uow.knowledge.update(
            _knowledge_payload(
                title="Patched title",
                created_at="2099-01-01T00:00:00Z",
                updated_at="2026-08-01T00:20:00Z",
            ),
            expected_version=1,
        )
        current = uow.knowledge.get(original.knowledge_id, original.memory_space_id)
        assert current is not None
        assert current.created_at == original.created_at


def test_concurrent_update_conflict_rejected(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with SQLiteUnitOfWork(db_path) as uow:
        migrations.apply_migrations(uow.connection)
        uow.memory_spaces.ensure("space-a", "hermes")

        uow.knowledge.add(_knowledge_payload())
        uow.knowledge.update(
            _knowledge_payload(
                knowledge_id="knowledge-1",
                title="first-commit",
                updated_at="2026-08-01T00:30:00Z",
            ),
            expected_version=1,
        )

        with pytest.raises(SQLiteKnowledgeConcurrencyError):
            uow.knowledge.update(
                _knowledge_payload(
                    knowledge_id="knowledge-1",
                    title="second-commit",
                    updated_at="2026-08-01T00:31:00Z",
                ),
                expected_version=1,
            )
