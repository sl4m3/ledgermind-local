"""SQLite repository for atom persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
import sqlite3
from typing import Iterable


def _normalize_message_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _to_json_array(values: Iterable[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)


def _from_json_array(raw: str) -> tuple[str, ...]:
    return tuple(json.loads(raw))


@dataclass(frozen=True, slots=True)
class Atom:
    """Atom persistence shape for local SQLite storage."""

    atom_id: str
    memory_space_id: str

    source_system: str
    source_instance_id: str
    source_profile_id: str
    source_session_id: str
    source_round_id: str
    source_round_key: str

    first_message_id: str | None
    final_message_id: str | None
    message_ids: tuple[str, ...]

    source_digest: str
    source_schema_version: int
    resolver_version: int

    extraction_host: str
    extraction_provider: str
    extraction_model: str
    extraction_prompt_version: int
    extraction_schema_version: int
    extraction_purpose: str

    title: str
    target: str
    statement: str
    rationale: str
    result: str
    artifacts: tuple[str, ...]
    content_digest: str

    supersedes_atom_id: str | None
    created_at: str


class SQLiteAtomRepository:
    """Repository for writing and reading atoms."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(self, atom: Atom) -> None:
        message_ids = _normalize_message_ids(atom.message_ids)
        artifacts = atom.artifacts

        self._connection.execute(
            """
            INSERT INTO atoms (
                atom_id,
                memory_space_id,
                source_system,
                source_instance_id,
                source_profile_id,
                source_session_id,
                source_round_id,
                source_round_key,
                first_message_id,
                final_message_id,
                message_ids_json,
                source_digest,
                source_schema_version,
                resolver_version,
                extraction_host,
                extraction_provider,
                extraction_model,
                extraction_prompt_version,
                extraction_schema_version,
                extraction_purpose,
                title,
                target,
                statement,
                rationale,
                result,
                artifacts_json,
                content_digest,
                supersedes_atom_id,
                created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                atom.atom_id,
                atom.memory_space_id,
                atom.source_system,
                atom.source_instance_id,
                atom.source_profile_id,
                atom.source_session_id,
                atom.source_round_id,
                atom.source_round_key,
                atom.first_message_id,
                atom.final_message_id,
                _to_json_array(message_ids),
                atom.source_digest,
                atom.source_schema_version,
                atom.resolver_version,
                atom.extraction_host,
                atom.extraction_provider,
                atom.extraction_model,
                atom.extraction_prompt_version,
                atom.extraction_schema_version,
                atom.extraction_purpose,
                atom.title,
                atom.target,
                atom.statement,
                atom.rationale,
                atom.result,
                _to_json_array(artifacts),
                atom.content_digest,
                atom.supersedes_atom_id,
                atom.created_at,
            ),
        )

    def get(self, atom_id: str, memory_space_id: str) -> Atom | None:
        row = self._connection.execute(
            """
            SELECT
                atom_id,
                memory_space_id,
                source_system,
                source_instance_id,
                source_profile_id,
                source_session_id,
                source_round_id,
                source_round_key,
                first_message_id,
                final_message_id,
                message_ids_json,
                source_digest,
                source_schema_version,
                resolver_version,
                extraction_host,
                extraction_provider,
                extraction_model,
                extraction_prompt_version,
                extraction_schema_version,
                extraction_purpose,
                title,
                target,
                statement,
                rationale,
                result,
                artifacts_json,
                content_digest,
                supersedes_atom_id,
                created_at
            FROM atoms
            WHERE atom_id = ? AND memory_space_id = ?
            LIMIT 1
            """,
            (atom_id, memory_space_id),
        ).fetchone()
        return self._row_to_atom(row) if row is not None else None

    def get_by_source_round(
        self,
        memory_space_id: str,
        source_round_key: str,
        *,
        extraction_prompt_version: int,
        extraction_schema_version: int,
    ) -> Atom | None:
        row = self._connection.execute(
            """
            SELECT
                atom_id,
                memory_space_id,
                source_system,
                source_instance_id,
                source_profile_id,
                source_session_id,
                source_round_id,
                source_round_key,
                first_message_id,
                final_message_id,
                message_ids_json,
                source_digest,
                source_schema_version,
                resolver_version,
                extraction_host,
                extraction_provider,
                extraction_model,
                extraction_prompt_version,
                extraction_schema_version,
                extraction_purpose,
                title,
                target,
                statement,
                rationale,
                result,
                artifacts_json,
                content_digest,
                supersedes_atom_id,
                created_at
            FROM atoms
            WHERE memory_space_id = ?
              AND source_round_key = ?
              AND extraction_prompt_version = ?
              AND extraction_schema_version = ?
            LIMIT 1
            """,
            (
                memory_space_id,
                source_round_key,
                extraction_prompt_version,
                extraction_schema_version,
            ),
        ).fetchone()
        return self._row_to_atom(row) if row is not None else None

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> Atom:
        return Atom(
            atom_id=row["atom_id"],
            memory_space_id=row["memory_space_id"],
            source_system=row["source_system"],
            source_instance_id=row["source_instance_id"],
            source_profile_id=row["source_profile_id"],
            source_session_id=row["source_session_id"],
            source_round_id=row["source_round_id"],
            source_round_key=row["source_round_key"],
            first_message_id=row["first_message_id"],
            final_message_id=row["final_message_id"],
            message_ids=_from_json_array(row["message_ids_json"]),
            source_digest=row["source_digest"],
            source_schema_version=row["source_schema_version"],
            resolver_version=row["resolver_version"],
            extraction_host=row["extraction_host"],
            extraction_provider=row["extraction_provider"],
            extraction_model=row["extraction_model"],
            extraction_prompt_version=row["extraction_prompt_version"],
            extraction_schema_version=row["extraction_schema_version"],
            extraction_purpose=row["extraction_purpose"],
            title=row["title"],
            target=row["target"],
            statement=row["statement"],
            rationale=row["rationale"],
            result=row["result"],
            artifacts=_from_json_array(row["artifacts_json"]),
            content_digest=row["content_digest"],
            supersedes_atom_id=row["supersedes_atom_id"],
            created_at=row["created_at"],
        )
