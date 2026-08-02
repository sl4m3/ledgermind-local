"""SQLite repository for knowledge items."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Knowledge:
    """Persistent knowledge row shape."""

    knowledge_id: str
    memory_space_id: str

    title: str
    target: str
    statement: str
    rationale: str
    phase: str
    version: int

    created_at: str
    updated_at: str
    superseded_by_id: str | None
    deleted_at: str | None


class SQLiteKnowledgeConcurrencyError(RuntimeError):
    """Raised when optimistic version check fails."""


class SQLiteKnowledgeRepository:
    """Repository for knowledge_items table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, knowledge: Knowledge) -> None:
        self._connection.execute(
            """
            INSERT INTO knowledge_items (
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id,
                deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                knowledge.knowledge_id,
                knowledge.memory_space_id,
                knowledge.title,
                knowledge.target,
                knowledge.statement,
                knowledge.rationale,
                knowledge.phase,
                knowledge.version,
                knowledge.created_at,
                knowledge.updated_at,
                knowledge.superseded_by_id,
                knowledge.deleted_at,
            ),
        )

    def get(self, knowledge_id: str, memory_space_id: str) -> Knowledge | None:
        row = self._connection.execute(
            """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id,
                deleted_at
            FROM knowledge_items
            WHERE knowledge_id = ? AND memory_space_id = ?
            LIMIT 1
            """,
            (knowledge_id, memory_space_id),
        ).fetchone()
        return self._row_to_knowledge(row) if row is not None else None

    def list_by_space(self, memory_space_id: str) -> list[Knowledge]:
        rows = self._connection.execute(
            """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id,
                deleted_at
            FROM knowledge_items
            WHERE memory_space_id = ?
            ORDER BY created_at ASC
            """,
            (memory_space_id,),
        ).fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    def list_current_by_space(self, memory_space_id: str) -> list[Knowledge]:
        rows = self._connection.execute(
            """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id,
                deleted_at
            FROM knowledge_items
            WHERE memory_space_id = ?
              AND superseded_by_id IS NULL
              AND deleted_at IS NULL
            ORDER BY created_at ASC
            """,
            (memory_space_id,),
        ).fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    def mark_deleted(self, knowledge_id: str, memory_space_id: str, *, deleted_at: str) -> None:
        row = self.get(knowledge_id, memory_space_id)
        if row is None:
            raise SQLiteKnowledgeConcurrencyError("knowledge not found")

        self._connection.execute(
            """
            UPDATE knowledge_items
            SET deleted_at = ?
            WHERE knowledge_id = ?
              AND memory_space_id = ?
            """,
            (deleted_at, knowledge_id, memory_space_id),
        )

    def mark_superseded(self, knowledge_id: str, memory_space_id: str, *, superseded_by_id: str) -> None:
        row = self.get(knowledge_id, memory_space_id)
        if row is None:
            raise SQLiteKnowledgeConcurrencyError("knowledge not found")

        self._connection.execute(
            """
            UPDATE knowledge_items
            SET superseded_by_id = ?
            WHERE knowledge_id = ?
              AND memory_space_id = ?
            """,
            (superseded_by_id, knowledge_id, memory_space_id),
        )

    def update(
        self,
        knowledge: Knowledge,
        *,
        expected_version: int,
    ) -> Knowledge:
        current = self.get(knowledge.knowledge_id, knowledge.memory_space_id)
        if current is None:
            raise SQLiteKnowledgeConcurrencyError("knowledge not found")

        cursor = self._connection.execute(
            """
            UPDATE knowledge_items
            SET title = ?,
                target = ?,
                statement = ?,
                rationale = ?,
                phase = ?,
                version = version + 1,
                updated_at = ?,
                superseded_by_id = ?,
                deleted_at = ?
            WHERE knowledge_id = ?
              AND memory_space_id = ?
              AND version = ?
            """,
            (
                knowledge.title,
                knowledge.target,
                knowledge.statement,
                knowledge.rationale,
                knowledge.phase,
                knowledge.updated_at,
                knowledge.superseded_by_id,
                knowledge.deleted_at,
                knowledge.knowledge_id,
                knowledge.memory_space_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise SQLiteKnowledgeConcurrencyError(
                f"knowledge '{knowledge.knowledge_id}' was modified concurrently"
            )

        return self._row(knowledge.knowledge_id, knowledge.memory_space_id)

    def _row(self, knowledge_id: str, memory_space_id: str) -> Knowledge:
        row = self._connection.execute(
            """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id,
                deleted_at
            FROM knowledge_items
            WHERE knowledge_id = ? AND memory_space_id = ?
            LIMIT 1
            """,
            (knowledge_id, memory_space_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("knowledge disappeared after update")
        return self._row_to_knowledge(row)

    @staticmethod
    def _row_to_knowledge(row: sqlite3.Row) -> Knowledge:
        return Knowledge(
            knowledge_id=row["knowledge_id"],
            memory_space_id=row["memory_space_id"],
            title=row["title"],
            target=row["target"],
            statement=row["statement"],
            rationale=row["rationale"],
            phase=row["phase"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by_id=row["superseded_by_id"],
            deleted_at=row["deleted_at"],
        )
