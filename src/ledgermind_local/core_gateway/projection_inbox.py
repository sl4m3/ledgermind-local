"""Durable Local inbox for Core-owned projection events."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from .projection_contracts import CoreProjectionEvent


@dataclass(frozen=True, slots=True)
class ProjectionInboxDelivery:
    event: CoreProjectionEvent
    projection_name: str
    attempts: int

    @property
    def event_id(self) -> str:
        return self.event.event_id


class CoreProjectionInbox:
    """Persist Core events before acknowledging them to the Core process."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def save_events(
        self,
        events: Iterable[CoreProjectionEvent],
        *,
        projection_names: Iterable[str],
    ) -> None:
        names = tuple(dict.fromkeys(name for name in projection_names if name))
        if not names:
            raise ValueError("at least one projection name is required")
        materialized = tuple(events)
        with self._connection:
            for event in materialized:
                existing = self._connection.execute(
                    """
                    SELECT memory_space_id, aggregate_id, event_type,
                           payload_json, occurred_at
                    FROM core_projection_events
                    WHERE event_id = ?
                    """,
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    stored = tuple(existing)
                    incoming = (
                        event.memory_space_id,
                        event.aggregate_id,
                        event.event_type,
                        event.payload_json,
                        event.occurred_at,
                    )
                    if stored != incoming:
                        raise ValueError(
                            f"Core event {event.event_id} conflicts with stored event"
                        )
                else:
                    self._connection.execute(
                        """
                        INSERT INTO core_projection_events (
                            event_id, memory_space_id, aggregate_id, event_type,
                            payload_json, occurred_at, received_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.memory_space_id,
                            event.aggregate_id,
                            event.event_type,
                            event.payload_json,
                            event.occurred_at,
                            _now(),
                        ),
                    )
                for projection_name in names:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO core_projection_deliveries (
                            projection_name, event_id, status, attempts, available_at
                        ) VALUES (?, ?, 'pending', 0, ?)
                        """,
                        (projection_name, event.event_id, _now()),
                    )

    def ready(
        self, projection_name: str, *, limit: int = 100
    ) -> list[ProjectionInboxDelivery]:
        if not projection_name:
            raise ValueError("projection_name must not be empty")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self._connection.execute(
            """
            SELECT
                e.event_id, e.memory_space_id, e.aggregate_id, e.event_type,
                e.payload_json, e.occurred_at, d.attempts
            FROM core_projection_deliveries d
            JOIN core_projection_events e ON e.event_id = d.event_id
            WHERE d.projection_name = ?
              AND d.status IN ('pending', 'failed')
              AND d.available_at <= ?
            ORDER BY e.occurred_at ASC, e.event_id ASC
            LIMIT ?
            """,
            (projection_name, _now(), limit),
        ).fetchall()
        return [
            ProjectionInboxDelivery(
                event=CoreProjectionEvent(
                    event_id=row[0],
                    memory_space_id=row[1],
                    aggregate_id=row[2],
                    event_type=row[3],
                    payload_json=row[4],
                    occurred_at=row[5],
                ),
                projection_name=projection_name,
                attempts=int(row[6]),
            )
            for row in rows
        ]

    def mark_processed(self, projection_name: str, event_id: str) -> None:
        with self._connection:
            updated = self._connection.execute(
                """
                UPDATE core_projection_deliveries
                SET status = 'processed', processed_at = ?, last_error = NULL
                WHERE projection_name = ? AND event_id = ?
                """,
                (_now(), projection_name, event_id),
            ).rowcount
            if updated != 1:
                raise KeyError(
                    f"unknown Core projection delivery: {projection_name}/{event_id}"
                )

    def mark_failed(
        self,
        projection_name: str,
        event_id: str,
        error: str,
        *,
        retry_after_seconds: int = 0,
    ) -> None:
        if not error.strip():
            raise ValueError("projection error must not be empty")
        with self._connection:
            updated = self._connection.execute(
                """
                UPDATE core_projection_deliveries
                SET status = 'failed', attempts = attempts + 1,
                    available_at = datetime('now', ?), last_error = ?
                WHERE projection_name = ? AND event_id = ?
                """,
                (
                    f"+{max(retry_after_seconds, 0)} seconds",
                    error[:500],
                    projection_name,
                    event_id,
                ),
            ).rowcount
            if updated != 1:
                raise KeyError(
                    f"unknown Core projection delivery: {projection_name}/{event_id}"
                )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["CoreProjectionInbox", "ProjectionInboxDelivery"]
