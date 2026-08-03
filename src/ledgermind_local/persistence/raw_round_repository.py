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
            SELECT raw_round_id, memory_space_id, source_system, source_instance_id,
                   source_profile_id, source_session_id, source_round_id, source_round_key,
                   capture_schema_version, adapter_version, payload_json, payload_digest,
                   started_at, completed_at, received_at, retention_expires_at
            FROM raw_rounds
            WHERE memory_space_id = ? AND source_round_key = ? AND capture_schema_version = ?
            """,
            (memory_space_id, source_round_key, capture_schema_version),
        ).fetchone()
        return self._round_from_row(row) if row is not None else None

    def get(self, raw_round_id: str) -> RawRoundRecord | None:
        row = self._connection.execute(
            "SELECT * FROM raw_rounds WHERE raw_round_id = ?",
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
                payload_json,
                payload_digest,
                started_at,
                completed_at,
                received_at,
                retention_expires_at,
            ),
        )
        result = self.get(raw_round_id)
        if result is None:
            raise RuntimeError("raw round insert did not produce a row")
        return result

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
            (job_id, raw_round_id, pipeline_version, normalizer_version, prompt_version, _now()),
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

    def claim_ready_job(self, *, worker_id: str) -> RoundProcessingJob | None:
        now = _now()
        row = self._connection.execute(
            """
            SELECT * FROM round_processing_jobs
            WHERE status IN ('received', 'retry_wait') AND available_at <= ?
            ORDER BY available_at, job_id
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        if row is None:
            return None
        updated = self._connection.execute(
            """
            UPDATE round_processing_jobs
            SET status = 'processing', claimed_at = ?, claimed_by = ?, attempts = attempts + 1
            WHERE job_id = ? AND status IN ('received', 'retry_wait')
            """,
            (now, worker_id, row["job_id"]),
        )
        if updated.rowcount != 1:
            return None
        claimed = self._connection.execute(
            "SELECT * FROM round_processing_jobs WHERE job_id = ?", (row["job_id"],)
        ).fetchone()
        return self._job_from_row(claimed) if claimed is not None else None

    def finish_job(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        retry_delay_seconds: float = 0,
    ) -> None:
        if status not in {"completed", "no_knowledge", "retry_wait", "failed"}:
            raise ValueError(f"unsupported processing status: {status}")
        now = datetime.now(timezone.utc)
        available_at = now + timedelta(seconds=max(retry_delay_seconds, 0))
        self._connection.execute(
            """
            UPDATE round_processing_jobs
            SET status = ?, completed_at = CASE WHEN ? IN ('completed', 'no_knowledge', 'failed') THEN ? ELSE NULL END,
                available_at = CASE WHEN ? = 'retry_wait' THEN ? ELSE available_at END,
                last_error = ?, claimed_at = NULL, claimed_by = NULL
            WHERE job_id = ?
            """,
            (
                status,
                status,
                now.isoformat(timespec="seconds"),
                status,
                available_at.isoformat(timespec="seconds"),
                error,
                job_id,
            ),
        )

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
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, raw_round_id, attempt_id, hypothesis_index, title, target,
                statement, rationale, result, artifacts_json, content_digest, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _now(),
            ),
        )

    def purge_expired(self, *, now: str | None = None, limit: int = 100) -> int:
        """Delete expired raw evidence only when no hypothesis references it."""

        cutoff = now or _now()
        bounded_limit = max(int(limit), 1)
        rows = self._connection.execute(
            """
            SELECT raw_round_id
            FROM raw_rounds
            WHERE retention_expires_at IS NOT NULL
              AND retention_expires_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM hypotheses
                  WHERE hypotheses.raw_round_id = raw_rounds.raw_round_id
              )
            ORDER BY retention_expires_at ASC
            LIMIT ?
            """,
            (cutoff, bounded_limit),
        ).fetchall()
        deleted = 0
        for row in rows:
            result = self._connection.execute(
                "DELETE FROM raw_rounds WHERE raw_round_id = ?",
                (row["raw_round_id"],),
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
        )
