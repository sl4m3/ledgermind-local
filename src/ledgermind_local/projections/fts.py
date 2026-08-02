"""SQLite FTS projection for knowledge search."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, ClassVar, cast

from ledgermind_core.ports import KnowledgeSearch, SearchHit

__all__ = [
    "KnowledgeFTSProjection",
    "SQLiteKnowledgeSearchAdapter",
]


_PROJECTION_NAME = "projections.search"
_PROJECTION_VERSION = 1


class _SafeTokenizer:
    _TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
    _STOP_TOKENS: ClassVar[set[str]] = {"and", "or", "not", "near"}

    @classmethod
    def tokenize(cls, query: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for token in cls._TOKEN_RE.findall(query):
            normalized = token.strip().lower()
            if not normalized or normalized in cls._STOP_TOKENS:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(normalized)
        return tokens


class KnowledgeFTSProjection:
    """Projector that maintains the ``knowledge_fts`` index from knowledge events."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def _table_exists(self, table_name: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name = ?
            LIMIT 1
            """,
            (table_name,),
        ).fetchone()
        return row is not None

    def _load_knowledge(self, *, knowledge_id: str, memory_space_id: str) -> sqlite3.Row | None:
        row = self._connection.execute(
            """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                superseded_by_id,
                deleted_at
            FROM knowledge_items
            WHERE knowledge_id = ?
              AND memory_space_id = ?
            LIMIT 1
            """,
            (knowledge_id, memory_space_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def _knowledge_in_index(self, *, knowledge_id: str, memory_space_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM knowledge_fts
            WHERE knowledge_id = ?
              AND memory_space_id = ?
            LIMIT 1
            """,
            (knowledge_id, memory_space_id),
        ).fetchone()
        return row is not None

    def _remove(self, knowledge_id: str, memory_space_id: str) -> bool:
        if not self._table_exists("knowledge_fts"):
            return False
        had_entry = self._knowledge_in_index(
            knowledge_id=knowledge_id,
            memory_space_id=memory_space_id,
        )
        self._connection.execute(
            """
            DELETE FROM knowledge_fts
            WHERE knowledge_id = ?
              AND memory_space_id = ?
            """,
            (knowledge_id, memory_space_id),
        )
        return had_entry

    def _upsert(self, knowledge_id: str, memory_space_id: str) -> bool:
        if not self._table_exists("knowledge_fts"):
            return False

        row = self._load_knowledge(knowledge_id=knowledge_id, memory_space_id=memory_space_id)
        if row is None:
            return self._remove(knowledge_id=knowledge_id, memory_space_id=memory_space_id)

        current = row["deleted_at"] is None and row["superseded_by_id"] is None
        if not current:
            return self._remove(knowledge_id=knowledge_id, memory_space_id=memory_space_id)

        self._remove(knowledge_id=knowledge_id, memory_space_id=memory_space_id)
        self._connection.execute(
            """
            INSERT INTO knowledge_fts (
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["knowledge_id"],
                row["memory_space_id"],
                row["title"],
                row["target"],
                row["statement"],
                row["rationale"],
            ),
        )
        return True

    @staticmethod
    def _coerce_string(value: Any) -> str | None:
        if isinstance(value, str) and value:
            return value
        return None

    @staticmethod
    def _extract_payload(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _event_knowledge_id(
        self,
        payload: dict[str, Any],
        aggregate_id: str,
    ) -> str | None:
        return (
            self._coerce_string(payload.get("knowledge_id"))
            or self._coerce_string(payload.get("aggregate_id"))
            or aggregate_id
            or None
        )

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str | None = None,
    ) -> bool:
        payload = self._extract_payload(payload_json)
        normalized_event = self._coerce_string(payload.get("event_type")) or event_type

        if normalized_event == "knowledge.created":
            knowledge_id = self._event_knowledge_id(payload, aggregate_id)
            if knowledge_id is None:
                return False
            if not memory_space_id:
                return False
            return self._upsert(knowledge_id, memory_space_id)

        if normalized_event == "knowledge.superseded":
            previous_id = self._coerce_string(payload.get("previous_knowledge_id"))
            next_id = self._coerce_string(payload.get("next_knowledge_id"))
            changed = False
            if previous_id is not None:
                changed = self._remove(previous_id, memory_space_id) or changed
            if next_id is not None:
                changed = self._upsert(next_id, memory_space_id) or changed
            return changed

        if normalized_event == "knowledge.deleted":
            knowledge_id = self._coerce_string(payload.get("knowledge_id")) or aggregate_id
            if knowledge_id is None:
                return False
            return self._remove(knowledge_id, memory_space_id)

        return False

    def _safe_match_query(self, query: str) -> str:
        tokens = _SafeTokenizer.tokenize(query)
        if not tokens:
            return ""
        return " ".join(tokens)

    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        if not self._table_exists("knowledge_fts"):
            return []

        safe_query = self._safe_match_query(query)
        if not safe_query or limit <= 0:
            return []

        try:
            rows = self._connection.execute(
                """
                SELECT
                    knowledge_id,
                    bm25(knowledge_fts) AS lexical_score
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                  AND memory_space_id = ?
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
            """
            SELECT knowledge_id, memory_space_id
            FROM knowledge_fts
            ORDER BY memory_space_id ASC, knowledge_id ASC
            """,
        ).fetchall()
        digest = hashlib.sha256()
        for row in rows:
            digest.update(row["memory_space_id"].encode("utf-8"))
            digest.update(b"\x00")
            digest.update(row["knowledge_id"].encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _write_projection_state(self, count: int) -> None:
        if not self._table_exists("projection_state"):
            return

        self._connection.execute(
            """
            INSERT INTO projection_state (
                projection_name,
                projection_version,
                item_count,
                checksum,
                rebuilt_at
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

    def rebuild(self, *, memory_space_id: str | None = None) -> int:
        if not self._table_exists("knowledge_fts"):
            return 0

        if memory_space_id is None:
            self._connection.execute("DELETE FROM knowledge_fts")
            current = self._connection.execute(
                """
                SELECT
                    knowledge_id,
                    memory_space_id,
                    title,
                    target,
                    statement,
                    rationale
                FROM knowledge_items
                WHERE superseded_by_id IS NULL
                  AND deleted_at IS NULL
                ORDER BY memory_space_id ASC, knowledge_id ASC
                """,
            ).fetchall()
            count = len(current)
        else:
            self._connection.execute(
                """
                DELETE FROM knowledge_fts
                WHERE memory_space_id = ?
                """,
                (memory_space_id,),
            )
            current = self._connection.execute(
                """
                SELECT
                    knowledge_id,
                    memory_space_id,
                    title,
                    target,
                    statement,
                    rationale
                FROM knowledge_items
                WHERE memory_space_id = ?
                  AND superseded_by_id IS NULL
                  AND deleted_at IS NULL
                ORDER BY knowledge_id ASC
                """,
                (memory_space_id,),
            ).fetchall()
            count = len(current)

        for row in current:
            self._connection.execute(
                """
                INSERT INTO knowledge_fts (
                    knowledge_id,
                    memory_space_id,
                    title,
                    target,
                    statement,
                    rationale
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["knowledge_id"],
                    row["memory_space_id"],
                    row["title"],
                    row["target"],
                    row["statement"],
                    row["rationale"],
                ),
            )

        if memory_space_id is not None:
            count = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS total FROM knowledge_fts WHERE memory_space_id = ?",
                    (memory_space_id,),
                ).fetchone()["total"]
            )
        else:
            count = int(
                self._connection.execute("SELECT COUNT(*) AS total FROM knowledge_fts").fetchone()[
                    "total"
                ]
            )

        self._write_projection_state(count)
        return count


class SQLiteKnowledgeSearchAdapter(KnowledgeSearch):
    """Search adapter backed by ``KnowledgeFTSProjection``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._projection = KnowledgeFTSProjection(connection=connection)

    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        return self._projection.search(memory_space_id, query, limit)
