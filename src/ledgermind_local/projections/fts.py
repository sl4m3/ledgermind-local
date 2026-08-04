"""SQLite FTS projection for Rust Core knowledge events."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from typing import ClassVar

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_DELETE,
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    ProjectionDeletePayload,
    ProjectionUpsertPayload,
)
from ledgermind_local.core_gateway.search_contracts import KnowledgeSearch, SearchHit

__all__ = ["KnowledgeFTSProjection", "SQLiteKnowledgeSearchAdapter"]

_PROJECTION_NAME = "projections.search"
_PROJECTION_VERSION = 1
_FTS_TABLE = "core_knowledge_fts"


class _SafeTokenizer:
    _TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
    _STOP_TOKENS: ClassVar[set[str]] = {"and", "or", "not", "near"}

    @classmethod
    def tokenize(cls, query: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for token in cls._TOKEN_RE.findall(query):
            normalized = token.strip().lower()
            if not normalized or normalized in cls._STOP_TOKENS or normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(normalized)
        return tokens


class KnowledgeFTSProjection:
    """Maintain the Local FTS projection from public Core payloads."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def _table_exists(self) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ? LIMIT 1
            """,
            (_FTS_TABLE,),
        ).fetchone()
        return row is not None

    def handle_core_event(self, event: CoreProjectionEvent) -> bool:
        """Apply a Core event without reading canonical Core rows."""

        parsed = event.parse_payload()
        if event.event_type == CORE_PROJECTION_UPSERT:
            if not isinstance(parsed, ProjectionUpsertPayload):
                raise TypeError("upsert event did not produce an upsert payload")
            if not self._table_exists():
                return False
            with self._connection:
                self._connection.execute(
                    f"DELETE FROM {_FTS_TABLE} WHERE knowledge_id = ? AND memory_space_id = ?",
                    (parsed.knowledge_id, parsed.memory_space_id),
                )
                self._connection.execute(
                    f"""
                    INSERT INTO {_FTS_TABLE} (
                        knowledge_id, memory_space_id, title, target, statement
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        parsed.knowledge_id,
                        parsed.memory_space_id,
                        parsed.title,
                        parsed.target,
                        parsed.statement,
                    ),
                )
            return True

        if event.event_type == CORE_PROJECTION_DELETE:
            if not isinstance(parsed, ProjectionDeletePayload):
                raise TypeError("delete event did not produce a delete payload")
            if not self._table_exists():
                return False
            with self._connection:
                row = self._connection.execute(
                    f"""
                    SELECT 1 FROM {_FTS_TABLE}
                    WHERE knowledge_id = ? AND memory_space_id = ? LIMIT 1
                    """,
                    (parsed.knowledge_id, parsed.memory_space_id),
                ).fetchone()
                self._connection.execute(
                    f"DELETE FROM {_FTS_TABLE} WHERE knowledge_id = ? AND memory_space_id = ?",
                    (parsed.knowledge_id, parsed.memory_space_id),
                )
            return row is not None

        raise ValueError(f"unsupported Core projection event type: {event.event_type}")

    @staticmethod
    def _safe_match_query(query: str) -> str:
        return " ".join(_SafeTokenizer.tokenize(query))

    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        return self.search_core(memory_space_id, query, limit)

    def search_core(
        self, memory_space_id: str, query: str, limit: int
    ) -> list[SearchHit]:
        if not self._table_exists():
            return []
        safe_query = self._safe_match_query(query)
        if not safe_query or limit <= 0:
            return []
        try:
            rows = self._connection.execute(
                f"""
                SELECT knowledge_id, bm25({_FTS_TABLE}) AS lexical_score
                FROM {_FTS_TABLE}
                WHERE {_FTS_TABLE} MATCH ? AND memory_space_id = ?
                ORDER BY lexical_score ASC
                LIMIT ?
                """,
                (safe_query, memory_space_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for row in rows:
            knowledge_id = row["knowledge_id"]
            if knowledge_id in seen:
                continue
            seen.add(knowledge_id)
            hits.append(
                SearchHit(
                    knowledge_id=knowledge_id,
                    lexical_score=float(row["lexical_score"]),
                    vector_score=None,
                )
            )
        return hits

    def _projection_checksum(self) -> str:
        rows = self._connection.execute(
            f"""
            SELECT knowledge_id, memory_space_id FROM {_FTS_TABLE}
            ORDER BY memory_space_id ASC, knowledge_id ASC
            """
        ).fetchall()
        digest = hashlib.sha256()
        for row in rows:
            digest.update(row["memory_space_id"].encode("utf-8"))
            digest.update(b"\x00")
            digest.update(row["knowledge_id"].encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def write_projection_state(self) -> None:
        """Record current event-derived projection metadata when the table exists."""

        if not self._table_exists():
            return
        row = self._connection.execute(f"SELECT COUNT(*) AS total FROM {_FTS_TABLE}").fetchone()
        count = int(row["total"]) if row is not None else 0
        if not self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'projection_state'"
        ).fetchone():
            return
        self._connection.execute(
            """
            INSERT INTO projection_state (
                projection_name, projection_version, item_count, checksum, rebuilt_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(projection_name) DO UPDATE SET
                projection_version = excluded.projection_version,
                item_count = excluded.item_count,
                checksum = excluded.checksum,
                rebuilt_at = excluded.rebuilt_at
            """,
            (
                self.projection_name,
                self.projection_version,
                count,
                self._projection_checksum(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


class SQLiteKnowledgeSearchAdapter(KnowledgeSearch):
    """Search adapter backed by the Local Core projection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._projection = KnowledgeFTSProjection(connection=connection)

    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        return self._projection.search_core(memory_space_id, query, limit)