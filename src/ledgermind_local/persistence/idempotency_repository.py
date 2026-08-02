"""SQLite repository for idempotency persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoredIdempotencyResult:
    """Persistent idempotency payload."""

    key: str
    request_hash: str
    response_json: str
    created_at: str
    expires_at: str | None
    memory_space_id: str = ""


class SQLiteIdempotencyRepository:
    """Repository wrapper over the ``idempotency_results`` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, result: StoredIdempotencyResult) -> None:
        self._connection.execute(
            """
            INSERT INTO idempotency_results (
                memory_space_id,
                idempotency_key,
                request_hash,
                response_json,
                created_at,
                expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.memory_space_id,
                result.key,
                result.request_hash,
                result.response_json,
                result.created_at,
                result.expires_at,
            ),
        )

    def get(
        self,
        memory_space_id: str,
        key: str | None = None,
        *,
        now: str | None = None,
    ) -> StoredIdempotencyResult | None:
        if key is None:
            key = memory_space_id
            memory_space_id = ""
        row = self._connection.execute(
            """
            SELECT
                idempotency_key,
                memory_space_id,
                request_hash,
                response_json,
                created_at,
                expires_at
            FROM idempotency_results
            WHERE memory_space_id = ? AND idempotency_key = ?
            LIMIT 1
            """,
            (memory_space_id, key),
        ).fetchone()
        if row is None:
            return None

        if now is not None and row["expires_at"] is not None and row["expires_at"] <= now:
            self._connection.execute(
                "DELETE FROM idempotency_results WHERE memory_space_id = ? AND idempotency_key = ?",
                (memory_space_id, key),
            )
            return None

        return StoredIdempotencyResult(
            key=row["idempotency_key"],
            request_hash=row["request_hash"],
            response_json=row["response_json"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            memory_space_id=row["memory_space_id"],
        )
