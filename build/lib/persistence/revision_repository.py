"""SQLite repository for knowledge revision snapshots."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeRevision:
    """Revision row shape for local storage."""

    revision_id: str
    knowledge_id: str
    version: int
    event_type: str
    snapshot_json: str
    cause_atom_id: str | None
    created_at: str


class SQLiteRevisionRepository:
    """Repository wrapper over the ``knowledge_revisions`` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, revision: KnowledgeRevision) -> None:
        self._connection.execute(
            """
            INSERT INTO knowledge_revisions (
                revision_id,
                knowledge_id,
                version,
                event_type,
                snapshot_json,
                cause_atom_id,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                revision.knowledge_id,
                revision.version,
                revision.event_type,
                revision.snapshot_json,
                revision.cause_atom_id,
                revision.created_at,
            ),
        )

    def list_for_knowledge(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> list[KnowledgeRevision]:
        rows = self._connection.execute(
            """
            SELECT r.revision_id,
                   r.knowledge_id,
                   r.version,
                   r.event_type,
                   r.snapshot_json,
                   r.cause_atom_id,
                   r.created_at
            FROM knowledge_revisions r
            JOIN knowledge_items k ON k.knowledge_id = r.knowledge_id
            WHERE r.knowledge_id = ? AND k.memory_space_id = ?
            ORDER BY r.version ASC
            """,
            (knowledge_id, memory_space_id),
        ).fetchall()
        return [self._row_to_revision(row) for row in rows]

    @staticmethod
    def _row_to_revision(row: sqlite3.Row) -> KnowledgeRevision:
        return KnowledgeRevision(
            revision_id=row["revision_id"],
            knowledge_id=row["knowledge_id"],
            version=row["version"],
            event_type=row["event_type"],
            snapshot_json=row["snapshot_json"],
            cause_atom_id=row["cause_atom_id"],
            created_at=row["created_at"],
        )
