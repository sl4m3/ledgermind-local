"""SQLite repository for knowledge-evidence relation persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeEvidence:
    """Evidence link row shape for local storage."""

    knowledge_id: str
    atom_id: str
    relation: str
    created_at: str


class SQLiteEvidenceRepository:
    """Repository wrapper over the ``knowledge_evidence`` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, evidence: KnowledgeEvidence) -> None:
        self._connection.execute(
            """
            INSERT INTO knowledge_evidence (
                knowledge_id,
                atom_id,
                relation,
                created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                evidence.knowledge_id,
                evidence.atom_id,
                evidence.relation,
                evidence.created_at,
            ),
        )

    def count_for_knowledge(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM knowledge_evidence e
            JOIN knowledge_items k ON k.knowledge_id = e.knowledge_id
            WHERE e.knowledge_id = ? AND k.memory_space_id = ?
            """,
            (knowledge_id, memory_space_id),
        ).fetchone()
        if row is None:
            return 0
        return int(row["total"])

    def list_atom_ids(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT e.atom_id
            FROM knowledge_evidence e
            JOIN knowledge_items k ON k.knowledge_id = e.knowledge_id
            WHERE e.knowledge_id = ? AND k.memory_space_id = ?
            ORDER BY e.created_at, e.atom_id
            """,
            (knowledge_id, memory_space_id),
        ).fetchall()
        return [row["atom_id"] for row in rows]

    def list_for_knowledge(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> list[KnowledgeEvidence]:
        rows = self._connection.execute(
            """
            SELECT e.knowledge_id, e.atom_id, e.relation, e.created_at
            FROM knowledge_evidence e
            JOIN knowledge_items k ON k.knowledge_id = e.knowledge_id
            WHERE e.knowledge_id = ? AND k.memory_space_id = ?
            ORDER BY e.created_at, e.atom_id
            """,
            (knowledge_id, memory_space_id),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> KnowledgeEvidence:
        return KnowledgeEvidence(
            knowledge_id=row["knowledge_id"],
            atom_id=row["atom_id"],
            relation=row["relation"],
            created_at=row["created_at"],
        )
