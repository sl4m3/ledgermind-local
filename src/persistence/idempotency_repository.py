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


class SQLiteIdempotencyRepository:
    """Repository wrapper over the ``idempotency_results`` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, result: StoredIdempotencyResult) -> None:
        self._connection.execute(
            """
            INSERT INTO idempotency_results (
                idempotency_key,
                request_hash,
                response_json,
                created_at,
                expires_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                result.key,
                result.request_hash,
                result.response_json,
                result.created_at,
                result.expires_at,
            ),
        )

    def get(self, key: str, *, now: str | None = None) -> StoredIdempotencyResult | None:
        row = self._connection.execute(
            """
            SELECT
                idempotency_key,
                request_hash,
                response_json,
                created_at,
                expires_at
            FROM idempotency_results
            WHERE idempotency_key = ?
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None

        if now is not None and row["expires_at"] is not None and row["expires_at"] <= now:
            self._connection.execute(
                "DELETE FROM idempotency_results WHERE idempotency_key = ?",
                (key,),
            )
            return None

        return StoredIdempotencyResult(
            key=row["idempotency_key"],
            request_hash=row["request_hash"],
            response_json=row["response_json"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
