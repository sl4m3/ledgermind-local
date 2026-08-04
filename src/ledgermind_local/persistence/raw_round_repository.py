"""SQLite repositories for immutable raw rounds and processing jobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RawRoundRecord:
    raw_round_id: str
    memory_space_id: str
    source_system: str
    source_instance_id: str
    source_profile_id: str
    source_session_id: str
    source_round_id: str
    source_round_key: str
    capture_schema_version: int
    adapter_version: str
    payload_json: str
    payload_digest: str
    started_at: str
    completed_at: str
    received_at: str
    retention_expires_at: str | None


@dataclass(frozen=True, slots=True)
class NormalizedRoundRecord:
    normalized_round_id: str
    raw_round_id: str
    normalizer_version: int
    payload_json: str
    payload_digest: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RoundProcessingJob:
    job_id: str
    raw_round_id: str
    pipeline_version: int
    normalizer_version: int
    prompt_version: int
    status: str
    attempts: int
    available_at: str
    claimed_at: str | None
    claimed_by: str | None
    completed_at: str | None
    last_error: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    lease_generation: int


@dataclass(frozen=True, slots=True)
class CoreCommandRecord:
    command_id: str
    command_type: str
    memory_space_id: str
    idempotency_key: str
    payload_json: str
    payload_digest: str
    status: str
    attempts: int
    available_at: str
    lease_expires_at: str | None
    claimed_by: str | None
    completed_at: str | None
    result_json: str | None
    last_error_code: str | None
    last_error_detail: str | None
    created_at: str


class SQLiteRawRoundRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get_by_capture_identity(
        self,
        memory_space_id: str,
        source_round_key: str,
        capture_schema_version: int,
    ) -> RawRoundRecord | None:
        row = self._connection.execute(
            """
            SELECT raw_rounds.raw_round_id, raw_rounds.memory_space_id,
                   raw_rounds.source_system, raw_rounds.source_instance_id,
                   raw_rounds.source_profile_id, raw_rounds.source_session_id,
                   raw_rounds.source_round_id, raw_rounds.source_round_key,
                   raw_rounds.capture_schema_version, raw_rounds.adapter_version,
                   COALESCE(raw_round_payloads.payload_json, raw_rounds.payload_json) AS payload_json,
                   raw_rounds.payload_digest, raw_rounds.started_at, raw_rounds.completed_at,
                   raw_rounds.received_at,
                   COALESCE(raw_round_payloads.retention_expires_at,
                            raw_rounds.retention_expires_at) AS retention_expires_at
            FROM raw_rounds
            LEFT JOIN raw_round_payloads
                ON raw_round_payloads.raw_round_id = raw_rounds.raw_round_id
            WHERE raw_rounds.memory_space_id = ?
              AND raw_rounds.source_round_key = ?
              AND raw_rounds.capture_schema_version = ?
            """,
            (memory_space_id, source_round_key, capture_schema_version),
        ).fetchone()
        return self._round_from_row(row) if row is not None else None

    def get(self, raw_round_id: str) -> RawRoundRecord | None:
        row = self._connection.execute(
            """
            SELECT raw_rounds.raw_round_id, raw_rounds.memory_space_id,
                   raw_rounds.source_system, raw_rounds.source_instance_id,
                   raw_rounds.source_profile_id, raw_rounds.source_session_id,
                   raw_rounds.source_round_id, raw_rounds.source_round_key,
                   raw_rounds.capture_schema_version, raw_rounds.adapter_version,
                   COALESCE(raw_round_payloads.payload_json, raw_rounds.payload_json) AS payload_json,
                   raw_rounds.payload_digest, raw_rounds.started_at, raw_rounds.completed_at,
                   raw_rounds.received_at,
                   COALESCE(raw_round_payloads.retention_expires_at,
                            raw_rounds.retention_expires_at) AS retention_expires_at
            FROM raw_rounds
            LEFT JOIN raw_round_payloads
                ON raw_round_payloads.raw_round_id = raw_rounds.raw_round_id
            WHERE raw_rounds.raw_round_id = ?
            """,
            (raw_round_id,),
        ).fetchone()
        return self._round_from_row(row) if row is not None else None

    def insert(
        self,
        *,
        raw_round_id: str,
        memory_space_id: str,
        source_system: str,
        source_instance_id: str,
        source_profile_id: str,
        source_session_id: str,
        source_round_id: str,
        source_round_key: str,
        capture_schema_version: int,
        adapter_version: str,
        payload_json: str,
        payload_digest: str,
        started_at: str,
        completed_at: str,
        retention_expires_at: str | None = None,
    ) -> RawRoundRecord:
        received_at = _now()
        self._connection.execute(
            """
            INSERT INTO raw_rounds (
                raw_round_id, memory_space_id, source_system, source_instance_id,
                source_profile_id, source_session_id, source_round_id, source_round_key,
                capture_schema_version, adapter_version, payload_json, payload_digest,
                started_at, completed_at, received_at, retention_expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_round_id,
                memory_space_id,
                source_system,
                source_instance_id,
                source_profile_id,
                source_session_id,
                source_round_id,
                source_round_key,
                capture_schema_version,
                adapter_version,
                "{}",
                payload_digest,
                started_at,
                completed_at,
                received_at,
                retention_expires_at,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO raw_round_payloads (
                raw_round_id, payload_json, payload_bytes, retention_expires_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                raw_round_id,
                payload_json,
                len(payload_json.encode("utf-8")),
                retention_expires_at,
            ),
        )
        result = self.get(raw_round_id)
        if result is None:
            raise RuntimeError("raw round insert did not produce a row")
        return result

    def store_normalized_round(
        self,
        *,
        raw_round_id: str,
        normalizer_version: int,
        payload_json: str,
        payload_digest: str,
    ) -> NormalizedRoundRecord:
        if normalizer_version < 1:
            raise ValueError("normalizer_version must be positive")
        normalized_round_id = f"{raw_round_id}:normalizer:{normalizer_version}"
        existing = self._connection.execute(
            """
            SELECT * FROM normalized_rounds
            WHERE raw_round_id = ? AND normalizer_version = ?
            """,
            (raw_round_id, normalizer_version),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["payload_digest"]) != payload_digest
                or str(existing["payload_json"]) != payload_json
            ):
                raise ValueError("normalized round digest changed for existing version")
            return self._normalized_round_from_row(existing)

        self._connection.execute(
            """
            INSERT INTO normalized_rounds (
                normalized_round_id, raw_round_id, normalizer_version,
                payload_json, payload_digest, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_round_id,
                raw_round_id,
                normalizer_version,
                payload_json,
                payload_digest,
                _now(),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM normalized_rounds WHERE normalized_round_id = ?",
            (normalized_round_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("normalized round insert did not produce a row")
        return self._normalized_round_from_row(row)

    def create_job(
        self,
        *,
        job_id: str,
        raw_round_id: str,
        pipeline_version: int,
        normalizer_version: int,
        prompt_version: int,
    ) -> RoundProcessingJob:
        self._connection.execute(
            """
            INSERT INTO round_processing_jobs (
                job_id, raw_round_id, pipeline_version, normalizer_version,
                prompt_version, status, attempts, available_at
            ) VALUES (?, ?, ?, ?, ?, 'received', 0, ?)
            """,
            (
                job_id,
                raw_round_id,
                pipeline_version,
                normalizer_version,
                prompt_version,
                _now(),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM round_processing_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("processing job insert did not produce a row")
        return self._job_from_row(row)

    def get_job_for_round(
        self,
        raw_round_id: str,
        pipeline_version: int,
        normalizer_version: int,
        prompt_version: int,
    ) -> RoundProcessingJob | None:
        row = self._connection.execute(
            """
            SELECT * FROM round_processing_jobs
            WHERE raw_round_id = ? AND pipeline_version = ?
              AND normalizer_version = ? AND prompt_version = ?
            """,
            (raw_round_id, pipeline_version, normalizer_version, prompt_version),
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def claim_ready_job(
        self,
        *,
        worker_id: str,
        lease_seconds: float = 300,
    ) -> RoundProcessingJob | None:
        now_datetime = datetime.now(timezone.utc)
        now = now_datetime.isoformat(timespec="seconds")
        lease_expires_at = (
            now_datetime + timedelta(seconds=max(float(lease_seconds), 1))
        ).isoformat()
        row = self._connection.execute(
            """
            SELECT * FROM round_processing_jobs
            WHERE (
                status IN ('received', 'retry_wait') AND available_at <= ?
            ) OR (
                status = 'processing'
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ?
            )
            ORDER BY available_at, job_id
            LIMIT 1
            """,
            (now, now),
        ).fetchone()
        if row is None:
            return None
        updated = self._connection.execute(
            """
            UPDATE round_processing_jobs
            SET status = 'processing', claimed_at = ?, claimed_by = ?, attempts = attempts + 1,
                lease_expires_at = ?, heartbeat_at = ?, lease_generation = lease_generation + 1
            WHERE job_id = ? AND (
                (status IN ('received', 'retry_wait') AND available_at <= ?)
                OR (
                    status = 'processing'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= ?
                )
            )
            """,
            (now, worker_id, lease_expires_at, now, row["job_id"], now, now),
        )
        if updated.rowcount != 1:
            return None
        claimed = self._connection.execute(
            "SELECT * FROM round_processing_jobs WHERE job_id = ?", (row["job_id"],)
        ).fetchone()
        return self._job_from_row(claimed) if claimed is not None else None

    def heartbeat(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_generation: int,
        lease_seconds: float = 300,
    ) -> bool:
        now_datetime = datetime.now(timezone.utc)
        now = now_datetime.isoformat(timespec="seconds")
        lease_expires_at = (
            now_datetime + timedelta(seconds=max(float(lease_seconds), 1))
        ).isoformat()
        updated = self._connection.execute(
            """
            UPDATE round_processing_jobs
            SET heartbeat_at = ?, lease_expires_at = ?
            WHERE job_id = ? AND status = 'processing' AND claimed_by = ?
              AND lease_generation = ? AND lease_expires_at > ?
            """,
            (now, lease_expires_at, job_id, worker_id, lease_generation, now),
        )
        return updated.rowcount == 1

    def finish_job(
        self,
        job_id: str,
        status: str,
        *,
        worker_id: str,
        lease_generation: int,
        error: str | None = None,
        retry_delay_seconds: float = 0,
    ) -> bool:
        if status not in {"completed", "no_knowledge", "retry_wait", "failed"}:
            raise ValueError(f"unsupported processing status: {status}")
        now = datetime.now(timezone.utc)
        available_at = now + timedelta(seconds=max(retry_delay_seconds, 0))
        updated = self._connection.execute(
            """
            UPDATE round_processing_jobs
            SET status = ?, completed_at = CASE WHEN ? IN ('completed', 'no_knowledge', 'failed') THEN ? ELSE NULL END,
                available_at = CASE WHEN ? = 'retry_wait' THEN ? ELSE available_at END,
                last_error = ?, claimed_at = NULL, claimed_by = NULL,
                lease_expires_at = NULL, heartbeat_at = NULL
            WHERE job_id = ?
              AND status = 'processing'
              AND claimed_by = ?
              AND lease_generation = ?
              AND lease_expires_at > ?
            """,
            (
                status,
                status,
                now.isoformat(timespec="seconds"),
                status,
                available_at.isoformat(timespec="seconds"),
                error,
                job_id,
                worker_id,
                lease_generation,
                now.isoformat(timespec="seconds"),
            ),
        )
        return updated.rowcount == 1

    def create_attempt(
        self,
        *,
        attempt_id: str,
        raw_round_id: str,
        job_id: str,
        pipeline_version: int,
        normalizer_version: int,
        provider: str,
        model: str,
        prompt_version: int,
        schema_version: int,
        started_at: str,
        completed_at: str | None = None,
        response_digest: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO hypothesis_attempts (
                attempt_id, raw_round_id, job_id, pipeline_version, normalizer_version,
                provider, model, prompt_version, schema_version, started_at, completed_at,
                response_digest, error_code, error_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                raw_round_id,
                job_id,
                pipeline_version,
                normalizer_version,
                provider,
                model,
                prompt_version,
                schema_version,
                started_at,
                completed_at,
                response_digest,
                error_code,
                error_detail,
            ),
        )

    def insert_hypothesis(
        self,
        *,
        hypothesis_id: str,
        raw_round_id: str,
        attempt_id: str,
        hypothesis_index: int,
        title: str,
        target: str,
        statement: str,
        rationale: str,
        result: str,
        artifacts_json: str,
        content_digest: str,
        status: str = "generated",
        core_command_id: str | None = None,
    ) -> None:
        if status not in {
            "generated",
            "queued_for_core",
            "accepted_by_core",
            "rejected_by_core",
        }:
            raise ValueError(f"unsupported hypothesis status: {status}")
        self._connection.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, raw_round_id, attempt_id, hypothesis_index, title, target,
                statement, rationale, result, artifacts_json, content_digest, status,
                core_command_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hypothesis_id,
                raw_round_id,
                attempt_id,
                hypothesis_index,
                title,
                target,
                statement,
                rationale,
                result,
                artifacts_json,
                content_digest,
                status,
                core_command_id,
                _now(),
            ),
        )

    def update_hypothesis_core_status(
        self,
        hypothesis_id: str,
        *,
        status: str,
        core_command_id: str | None = None,
    ) -> bool:
        if status not in {
            "generated",
            "queued_for_core",
            "accepted_by_core",
            "rejected_by_core",
        }:
            raise ValueError(f"unsupported hypothesis status: {status}")
        updated = self._connection.execute(
            """
            UPDATE hypotheses
            SET status = ?, core_command_id = COALESCE(?, core_command_id)
            WHERE hypothesis_id = ?
            """,
            (status, core_command_id, hypothesis_id),
        )
        return updated.rowcount == 1

    def create_core_command(
        self,
        *,
        command_id: str,
        command_type: str,
        memory_space_id: str,
        idempotency_key: str,
        payload_json: str,
        payload_digest: str,
        available_at: str | None = None,
    ) -> CoreCommandRecord:
        if not command_type.strip():
            raise ValueError("command_type must not be empty")
        if not payload_digest.startswith("sha256:"):
            raise ValueError("payload_digest must use sha256 prefix")
        existing = self._connection.execute(
            """
            SELECT * FROM core_commands
            WHERE memory_space_id = ? AND idempotency_key = ?
            """,
            (memory_space_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["command_type"]) != command_type
                or str(existing["payload_json"]) != payload_json
                or str(existing["payload_digest"]) != payload_digest
            ):
                raise ValueError("core command payload conflict")
            return self._core_command_from_row(existing)

        created_at = _now()
        self._connection.execute(
            """
            INSERT INTO core_commands (
                command_id, command_type, memory_space_id, idempotency_key,
                payload_json, payload_digest, status, attempts, available_at,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (
                command_id,
                command_type,
                memory_space_id,
                idempotency_key,
                payload_json,
                payload_digest,
                available_at or created_at,
                created_at,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM core_commands WHERE command_id = ?", (command_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("core command insert did not produce a row")
        return self._core_command_from_row(row)

    def get_core_command(self, command_id: str) -> CoreCommandRecord | None:
        row = self._connection.execute(
            "SELECT * FROM core_commands WHERE command_id = ?", (command_id,)
        ).fetchone()
        return self._core_command_from_row(row) if row is not None else None

    def claim_core_command(
        self,
        *,
        worker_id: str,
        lease_seconds: float = 300,
        now: str | None = None,
    ) -> CoreCommandRecord | None:
        now_datetime = (
            datetime.fromisoformat(now)
            if now is not None
            else datetime.now(timezone.utc)
        )
        if now_datetime.tzinfo is None:
            now_datetime = now_datetime.replace(tzinfo=timezone.utc)
        now_value = now_datetime.isoformat(timespec="seconds")
        lease_expires_at = (
            now_datetime + timedelta(seconds=max(float(lease_seconds), 1))
        ).isoformat(timespec="seconds")
        row = self._connection.execute(
            """
            SELECT * FROM core_commands
            WHERE (
                status IN ('pending', 'retry_wait') AND available_at <= ?
            ) OR (
                status = 'delivering'
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ?
            )
            ORDER BY available_at, command_id
            LIMIT 1
            """,
            (now_value, now_value),
        ).fetchone()
        if row is None:
            return None
        updated = self._connection.execute(
            """
            UPDATE core_commands
            SET status = 'delivering', attempts = attempts + 1,
                lease_expires_at = ?, claimed_by = ?, completed_at = NULL
            WHERE command_id = ? AND (
                (status IN ('pending', 'retry_wait') AND available_at <= ?)
                OR (
                    status = 'delivering'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= ?
                )
            )
            """,
            (lease_expires_at, worker_id, row["command_id"], now_value, now_value),
        )
        if updated.rowcount != 1:
            return None
        claimed = self._connection.execute(
            "SELECT * FROM core_commands WHERE command_id = ?", (row["command_id"],)
        ).fetchone()
        return self._core_command_from_row(claimed) if claimed is not None else None

    def finish_core_command(
        self,
        command_id: str,
        *,
        worker_id: str,
        status: str,
        result_json: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        retry_delay_seconds: float = 0,
    ) -> bool:
        if status not in {"completed", "retry_wait", "rejected", "failed"}:
            raise ValueError(f"unsupported core command status: {status}")
        now_datetime = datetime.now(timezone.utc)
        now_value = now_datetime.isoformat(timespec="seconds")
        available_at = (
            now_datetime + timedelta(seconds=max(float(retry_delay_seconds), 0))
        ).isoformat(timespec="seconds")
        updated = self._connection.execute(
            """
            UPDATE core_commands
            SET status = ?,
                available_at = CASE WHEN ? = 'retry_wait' THEN ? ELSE available_at END,
                lease_expires_at = NULL,
                claimed_by = NULL,
                completed_at = CASE WHEN ? = 'retry_wait' THEN NULL ELSE ? END,
                result_json = COALESCE(?, result_json),
                last_error_code = ?,
                last_error_detail = ?
            WHERE command_id = ? AND status = 'delivering' AND claimed_by = ?
            """,
            (
                status,
                status,
                available_at,
                status,
                now_value,
                result_json,
                error_code,
                error_detail,
                command_id,
                worker_id,
            ),
        )
        return updated.rowcount == 1

    def purge_expired(self, *, now: str | None = None, limit: int = 100) -> int:
        """Clear expired payloads while preserving raw provenance and derivatives."""

        cutoff = now or _now()
        bounded_limit = max(int(limit), 1)
        rows = self._connection.execute(
            """
            SELECT raw_round_id
            FROM raw_round_payloads
            WHERE deleted_at IS NULL
              AND retention_expires_at IS NOT NULL
              AND retention_expires_at <= ?
            ORDER BY retention_expires_at ASC
            LIMIT ?
            """,
            (cutoff, bounded_limit),
        ).fetchall()
        deleted = 0
        for row in rows:
            result = self._connection.execute(
                """
                UPDATE raw_round_payloads
                SET payload_json = '{}', payload_bytes = 0, deleted_at = ?
                WHERE raw_round_id = ? AND deleted_at IS NULL
                """,
                (cutoff, row["raw_round_id"]),
            )
            deleted += max(result.rowcount, 0)
        return deleted

    def _round_from_row(self, row: sqlite3.Row) -> RawRoundRecord:
        return RawRoundRecord(
            raw_round_id=str(row["raw_round_id"]),
            memory_space_id=str(row["memory_space_id"]),
            source_system=str(row["source_system"]),
            source_instance_id=str(row["source_instance_id"]),
            source_profile_id=str(row["source_profile_id"]),
            source_session_id=str(row["source_session_id"]),
            source_round_id=str(row["source_round_id"]),
            source_round_key=str(row["source_round_key"]),
            capture_schema_version=int(row["capture_schema_version"]),
            adapter_version=str(row["adapter_version"]),
            payload_json=str(row["payload_json"]),
            payload_digest=str(row["payload_digest"]),
            started_at=str(row["started_at"]),
            completed_at=str(row["completed_at"]),
            received_at=str(row["received_at"]),
            retention_expires_at=row["retention_expires_at"],
        )

    @staticmethod
    def _normalized_round_from_row(row: sqlite3.Row) -> NormalizedRoundRecord:
        return NormalizedRoundRecord(
            normalized_round_id=str(row["normalized_round_id"]),
            raw_round_id=str(row["raw_round_id"]),
            normalizer_version=int(row["normalizer_version"]),
            payload_json=str(row["payload_json"]),
            payload_digest=str(row["payload_digest"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> RoundProcessingJob:
        return RoundProcessingJob(
            job_id=str(row["job_id"]),
            raw_round_id=str(row["raw_round_id"]),
            pipeline_version=int(row["pipeline_version"]),
            normalizer_version=int(row["normalizer_version"]),
            prompt_version=int(row["prompt_version"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            available_at=str(row["available_at"]),
            claimed_at=row["claimed_at"],
            claimed_by=row["claimed_by"],
            completed_at=row["completed_at"],
            last_error=row["last_error"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            lease_generation=int(row["lease_generation"]),
        )

    @staticmethod
    def _core_command_from_row(row: sqlite3.Row) -> CoreCommandRecord:
        return CoreCommandRecord(
            command_id=str(row["command_id"]),
            command_type=str(row["command_type"]),
            memory_space_id=str(row["memory_space_id"]),
            idempotency_key=str(row["idempotency_key"]),
            payload_json=str(row["payload_json"]),
            payload_digest=str(row["payload_digest"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            available_at=str(row["available_at"]),
            lease_expires_at=row["lease_expires_at"],
            claimed_by=row["claimed_by"],
            completed_at=row["completed_at"],
            result_json=row["result_json"],
            last_error_code=row["last_error_code"],
            last_error_detail=row["last_error_detail"],
            created_at=str(row["created_at"]),
        )
