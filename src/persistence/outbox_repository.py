"""SQLite repository for durable outbox and projection deliveries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    """Persistent outbox event representation."""

    event_id: str
    event_type: str
    aggregate_id: str
    memory_space_id: str
    payload_json: str
    occurred_at: str
    available_at: str
    attempts: int
    claimed_at: str | None
    claimed_by: str | None
    processed_at: str | None
    last_error: str | None


class SQLiteOutboxRepository:
    """Repository wrapper over ``outbox_events`` and ``projection_deliveries``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(
        self,
        event: OutboxEvent,
        *,
        projection_names: tuple[str, ...],
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO outbox_events (
                event_id,
                event_type,
                aggregate_id,
                memory_space_id,
                payload_json,
                occurred_at,
                attempts,
                available_at,
                claimed_at,
                claimed_by,
                processed_at,
                last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.aggregate_id,
                event.memory_space_id,
                event.payload_json,
                event.occurred_at,
                event.attempts,
                event.available_at,
                event.claimed_at,
                event.claimed_by,
                event.processed_at,
                event.last_error,
            ),
        )

        seen: set[str] = set()
        for projection_name in projection_names:
            if projection_name in seen:
                continue
            seen.add(projection_name)
            self._connection.execute(
                """
                INSERT INTO projection_deliveries (
                    projection_name,
                    event_id,
                    attempts,
                    available_at,
                    claimed_at,
                    claimed_by,
                    processed_at,
                    last_error
                ) VALUES (?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    projection_name,
                    event.event_id,
                    event.available_at,
                    event.claimed_at,
                    event.claimed_by,
                    event.processed_at,
                    event.last_error,
                ),
            )

    def list_ready(
        self,
        projection_name: str,
        *,
        now: str,
    ) -> list[OutboxEvent]:
        rows = self._connection.execute(
            """
            SELECT
                o.event_id,
                o.event_type,
                o.aggregate_id,
                o.memory_space_id,
                o.payload_json,
                o.occurred_at,
                o.attempts,
                o.available_at,
                d.claimed_at,
                d.claimed_by,
                o.processed_at,
                d.last_error
            FROM outbox_events o
            JOIN projection_deliveries d ON d.event_id = o.event_id
            WHERE d.projection_name = ?
              AND o.processed_at IS NULL
              AND d.processed_at IS NULL
              AND d.claimed_at IS NULL
              AND d.available_at <= ?
            ORDER BY d.available_at ASC, o.occurred_at ASC
            """,
            (projection_name, now),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def acquire_next(
        self,
        projection_name: str,
        worker_id: str,
        now: str,
        *,
        stale_claim_before: str,
    ) -> OutboxEvent | None:
        self._release_stale_claims(projection_name, stale_claim_before)
        row = self._connection.execute(
            """
            SELECT
                o.event_id,
                o.event_type,
                o.aggregate_id,
                o.memory_space_id,
                o.payload_json,
                o.occurred_at,
                o.attempts,
                o.available_at,
                d.claimed_at,
                d.claimed_by,
                o.processed_at,
                d.last_error
            FROM outbox_events o
            JOIN projection_deliveries d ON d.event_id = o.event_id
            WHERE d.projection_name = ?
              AND o.processed_at IS NULL
              AND d.processed_at IS NULL
              AND d.claimed_at IS NULL
              AND d.available_at <= ?
            ORDER BY d.available_at ASC, o.occurred_at ASC
            LIMIT 1
            """,
            (projection_name, now),
        ).fetchone()
        if row is None:
            return None

        event_id = row["event_id"]
        cursor = self._connection.execute(
            """
            UPDATE projection_deliveries
            SET claimed_at = ?, claimed_by = ?
            WHERE projection_name = ? AND event_id = ? AND claimed_at IS NULL
            """,
            (now, worker_id, projection_name, event_id),
        )
        if cursor.rowcount == 0:
            return None

        return OutboxEvent(
            event_id=row["event_id"],
            event_type=row["event_type"],
            aggregate_id=row["aggregate_id"],
            memory_space_id=row["memory_space_id"],
            payload_json=row["payload_json"],
            occurred_at=row["occurred_at"],
            available_at=row["available_at"],
            attempts=row["attempts"],
            claimed_at=now,
            claimed_by=worker_id,
            processed_at=row["processed_at"],
            last_error=row["last_error"],
        )

    def mark_processed(
        self,
        projection_name: str,
        event_id: str,
        *,
        processed_at: str,
    ) -> bool:
        cursor = self._connection.execute(
            """
            UPDATE projection_deliveries
            SET processed_at = ?,
                claimed_at = NULL,
                claimed_by = NULL
            WHERE projection_name = ?
              AND event_id = ?
              AND processed_at IS NULL
            """,
            (processed_at, projection_name, event_id),
        )
        if cursor.rowcount == 0:
            return False

        remaining = self._connection.execute(
            """
            SELECT COUNT(*) AS pending
            FROM projection_deliveries
            WHERE event_id = ?
              AND processed_at IS NULL
            """,
            (event_id,),
        ).fetchone()

        if remaining is not None and int(remaining["pending"]) == 0:
            self._connection.execute(
                """
                UPDATE outbox_events
                SET processed_at = ?,
                    last_error = NULL
                WHERE event_id = ?
                """,
                (processed_at, event_id),
            )

        return True

    def mark_failed(
        self,
        projection_name: str,
        event_id: str,
        *,
        available_at: str,
        last_error: str | None,
    ) -> int:
        cursor = self._connection.execute(
            """
            UPDATE projection_deliveries
            SET attempts = attempts + 1,
                claimed_at = NULL,
                claimed_by = NULL,
                available_at = ?,
                last_error = ?
            WHERE projection_name = ?
              AND event_id = ?
              AND processed_at IS NULL
            """,
            (available_at, last_error, projection_name, event_id),
        )
        if cursor.rowcount == 0:
            return 0

        attempts = self._connection.execute(
            """
            SELECT attempts
            FROM projection_deliveries
            WHERE projection_name = ?
              AND event_id = ?
            """,
            (projection_name, event_id),
        ).fetchone()

        self._connection.execute(
            """
            UPDATE outbox_events
            SET attempts = attempts + 1,
                last_error = ?
            WHERE event_id = ?
            """,
            (last_error, event_id),
        )
        return int(attempts["attempts"]) if attempts is not None else 0

    def _release_stale_claims(self, projection_name: str, stale_claim_before: str) -> None:
        self._connection.execute(
            """
            UPDATE projection_deliveries
            SET claimed_at = NULL,
                claimed_by = NULL
            WHERE projection_name = ?
              AND processed_at IS NULL
              AND claimed_at IS NOT NULL
              AND claimed_at < ?
            """,
            (projection_name, stale_claim_before),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> OutboxEvent:
        return OutboxEvent(
            event_id=row["event_id"],
            event_type=row["event_type"],
            aggregate_id=row["aggregate_id"],
            memory_space_id=row["memory_space_id"],
            payload_json=row["payload_json"],
            occurred_at=row["occurred_at"],
            available_at=row["available_at"],
            attempts=row["attempts"],
            claimed_at=row["claimed_at"],
            claimed_by=row["claimed_by"],
            processed_at=row["processed_at"],
            last_error=row["last_error"],
        )
