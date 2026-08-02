"""Validation for a canonical LedgerMind SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "schema_migrations": ("version", "name"),
    "memory_spaces": ("memory_space_id", "source_client"),
    "atoms": (
        "atom_id",
        "memory_space_id",
        "source_round_key",
        "source_digest",
        "content_digest",
    ),
    "knowledge_items": ("knowledge_id", "memory_space_id", "phase", "version"),
    "knowledge_evidence": ("knowledge_id", "atom_id", "relation"),
    "knowledge_revisions": ("revision_id", "knowledge_id", "snapshot_json"),
    "idempotency_results": ("memory_space_id", "idempotency_key", "request_hash"),
    "outbox_events": ("event_id", "event_type", "memory_space_id"),
    "projection_deliveries": ("event_id", "projection_name", "processed_at"),
}


def validate_temp_database(path: Path) -> tuple[bool, list[str]]:
    """Check integrity and canonical tables without requiring migration-only tables."""

    if not path.exists():
        return False, ["temp_database_missing"]
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    messages: list[str] = []
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table, columns in _REQUIRED_TABLES.items():
            if table not in tables:
                messages.append(f"required_table_missing:{table}")
                continue
            actual_columns = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column in columns:
                if column not in actual_columns:
                    messages.append(f"required_column_missing:{table}:{column}")

        if not messages:
            fk_check = connection.execute("PRAGMA foreign_key_check").fetchall()
            if fk_check:
                messages.extend(
                    f"foreign_key_violation:{row[0]}:{row[1]}:{row[2]}"
                    for row in fk_check
                )

            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                messages.append(
                    f"integrity_check_failed:{integrity[0] if integrity else 'no result'}"
                )

            invalid_digests = connection.execute(
                """
                SELECT atom_id
                FROM atoms
                WHERE source_digest NOT LIKE 'sha256:%'
                   OR content_digest NOT LIKE 'sha256:%'
                LIMIT 20
                """
            ).fetchall()
            messages.extend(f"noncanonical_digest:{row[0]}" for row in invalid_digests)

            orphan_evidence = connection.execute(
                """
                SELECT e.knowledge_id, e.atom_id
                FROM knowledge_evidence e
                LEFT JOIN knowledge_items k ON k.knowledge_id = e.knowledge_id
                LEFT JOIN atoms a ON a.atom_id = e.atom_id
                WHERE k.knowledge_id IS NULL OR a.atom_id IS NULL
                LIMIT 20
                """
            ).fetchall()
            messages.extend(
                f"orphan_evidence:{row[0]}:{row[1]}" for row in orphan_evidence
            )
    finally:
        connection.close()
    return not messages, messages