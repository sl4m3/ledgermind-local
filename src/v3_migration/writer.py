from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v3_migration.models import LegacyRecord, MigrationManifest
from v3_migration.reader import read_legacy_storage
from v3_migration.validator import validate_temp_database


def _open_temp_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS legacy_id_map (
            migration_id TEXT NOT NULL,
            legacy_fid TEXT NOT NULL,
            atom_id TEXT,
            knowledge_id TEXT,
            action TEXT NOT NULL,
            warnings_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (migration_id, legacy_fid)
        )
        """
    )
    return connection


def _apply_database_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS atoms (
            atom_id TEXT PRIMARY KEY,
            memory_space_id TEXT NOT NULL,
            source_system TEXT NOT NULL,
            source_instance_id TEXT NOT NULL,
            source_profile_id TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_round_id TEXT NOT NULL,
            source_digest TEXT NOT NULL,
            statement TEXT NOT NULL DEFAULT '',
            title TEXT,
            target TEXT,
            rationale TEXT,
            artifacts TEXT NOT NULL DEFAULT '[]',
            supersedes TEXT NOT NULL DEFAULT '[]',
            extraction_host TEXT NOT NULL DEFAULT '',
            extraction_provider TEXT NOT NULL DEFAULT '',
            extraction_model TEXT NOT NULL DEFAULT '',
            prompt_version INTEGER NOT NULL DEFAULT 1,
            schema_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_items (
            knowledge_id TEXT PRIMARY KEY,
            memory_space_id TEXT NOT NULL,
            atom_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            target TEXT NOT NULL DEFAULT '',
            statement TEXT NOT NULL DEFAULT '',
            rationale TEXT NOT NULL DEFAULT '',
            phase TEXT NOT NULL DEFAULT 'PATTERN',
            vitality TEXT NOT NULL DEFAULT 'ACTIVE',
            status TEXT NOT NULL DEFAULT 'active',
            legacy_status TEXT,
            artifacts TEXT NOT NULL DEFAULT '[]',
            supersedes TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (atom_id) REFERENCES atoms(atom_id)
        )
        """
    )


def write_temp_migration(
    *,
    records: list[LegacyRecord],
    destination: Path,
    migration_id: str,
    apply: bool,
) -> tuple[list[MigrationManifest], list[str]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    connection = _open_temp_database(destination)
    manifests: list[MigrationManifest] = []
    warnings: list[str] = []
    try:
        _apply_database_migrations(connection)
        for record in records:
            mapped = {}
            try:
                from v3_migration.mapper import map_record  # local import to avoid cycles
                mapped = map_record(record)
            except Exception as exc:  # pragma: no cover - defensive guard
                warnings.append(f"map_failed:{record.fid}:{exc}")
                manifests.append(
                    MigrationManifest(
                        migration_id=migration_id,
                        legacy_fid=record.fid,
                        action="skipped",
                        warnings_json=json.dumps([f"map_failed:{exc}"], ensure_ascii=False),
                    )
                )
                continue
            atom_id = f"legacy:{migration_id}:{record.fid}"
            knowledge_id = f"legacy:{migration_id}:{record.fid}:knowledge"
            now = __import__("datetime").datetime.now(timezone.utc).isoformat(timespec="seconds")
            if not apply:
                manifests.append(
                    MigrationManifest(
                        migration_id=migration_id,
                        legacy_fid=record.fid,
                        atom_id=atom_id,
                        knowledge_id=knowledge_id,
                        action="planned",
                        warnings_json=json.dumps(mapped.get("warnings", []), ensure_ascii=False),
                    )
                )
                continue
            connection.execute(
                """
                INSERT INTO atoms (
                    atom_id, memory_space_id, source_system, source_instance_id,
                    source_profile_id, source_session_id, source_round_id, source_digest,
                    statement, title, target, rationale, artifacts, supersedes,
                    extraction_host, extraction_provider, extraction_model,
                    prompt_version, schema_version, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    atom_id,
                    mapped["memory_space_id"],
                    mapped["source_system"],
                    mapped["source_instance_id"],
                    mapped["source_profile_id"],
                    mapped["source_session_id"],
                    mapped["source_round_id"],
                    mapped["source_digest"],
                    mapped.get("statement", ""),
                    mapped.get("title"),
                    mapped.get("target"),
                    mapped.get("rationale"),
                    json.dumps(mapped.get("artifacts", []), ensure_ascii=False),
                    json.dumps(mapped.get("supersedes", []), ensure_ascii=False),
                    mapped["extraction_host"],
                    mapped["extraction_provider"],
                    mapped["extraction_model"],
                    mapped["prompt_version"],
                    mapped["schema_version"],
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO knowledge_items (
                    knowledge_id, memory_space_id, atom_id, title, target,
                    statement, rationale, phase, vitality, status, legacy_status,
                    artifacts, supersedes, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    knowledge_id,
                    mapped["memory_space_id"],
                    atom_id,
                    mapped.get("title") or "",
                    mapped.get("target") or "",
                    mapped.get("statement", ""),
                    mapped.get("rationale", ""),
                    mapped["phase"],
                    mapped["vitality"],
                    "active",
                    mapped.get("legacy_status"),
                    json.dumps(mapped.get("artifacts", []), ensure_ascii=False),
                    json.dumps(mapped.get("supersedes", []), ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO legacy_id_map (migration_id, legacy_fid, atom_id, knowledge_id, action, warnings_json) VALUES (?,?,?,?,?,?)",
                (
                    migration_id,
                    record.fid,
                    atom_id,
                    knowledge_id,
                    "migrated",
                    json.dumps(mapped.get("warnings", []), ensure_ascii=False),
                ),
            )
            manifests.append(
                MigrationManifest(
                    migration_id=migration_id,
                    legacy_fid=record.fid,
                    atom_id=atom_id,
                    knowledge_id=knowledge_id,
                    action="migrated",
                    warnings_json=json.dumps(mapped.get("warnings", []), ensure_ascii=False),
                )
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return manifests, warnings
