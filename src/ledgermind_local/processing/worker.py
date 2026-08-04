"""Crash-tolerant RawRound processing worker."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import Self

from ..persistence import RawRoundRecord, SQLiteUnitOfWork
from ..scheduler.guarded_loop import GuardedWorkerLoop
from ..scheduler.worker_state import WorkerState
from .generator import HypothesisCandidate, HypothesisGenerator
from .models import NormalizedRound
from .normalizer import normalize_raw_round, redact_text, redact_value

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    job_id: str
    raw_round_id: str
    status: str
    hypothesis_ids: tuple[str, ...] = ()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest_candidate(draft: HypothesisCandidate) -> str:
    material = {
        "title": draft.title,
        "target": draft.target,
        "statement": draft.statement,
        "rationale": draft.rationale,
        "result": draft.result,
        "artifacts": list(draft.artifacts),
        "source_event_ids": list(draft.source_event_ids),
    }
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _response_digest(drafts: Sequence[HypothesisCandidate]) -> str:
    material = [
        {
            "title": draft.title,
            "target": draft.target,
            "statement": draft.statement,
            "rationale": draft.rationale,
            "result": draft.result,
            "artifacts": list(draft.artifacts),
            "source_event_ids": list(draft.source_event_ids),
        }
        for draft in drafts
    ]
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalized_round_payload(normalized_round: NormalizedRound) -> str:
    payload = {
        "memory_space_id": normalized_round.memory_space_id,
        "source_system": normalized_round.source_system,
        "source_instance_id": normalized_round.source_instance_id,
        "source_profile_id": normalized_round.source_profile_id,
        "source_session_id": normalized_round.source_session_id,
        "source_round_id": normalized_round.source_round_id,
        "started_at": normalized_round.started_at,
        "completed_at": normalized_round.completed_at,
        "user_text": normalized_round.user_text,
        "assistant_text": normalized_round.assistant_text,
        "transcript": normalized_round.transcript,
        "tool_interactions": [
            {
                "tool_call_id": interaction.tool_call_id,
                "tool_name": interaction.tool_name,
                "arguments_json": interaction.arguments_json,
                "result_text": interaction.result_text,
                "result_json": interaction.result_json,
                "status": interaction.status,
                "error_text": interaction.error_text,
                "source_call_event_id": interaction.source_call_event_id,
                "source_result_event_id": interaction.source_result_event_id,
            }
            for interaction in normalized_round.tool_interactions
        ],
        "normalized_digest": normalized_round.normalized_digest,
        "source_event_ids": list(normalized_round.source_event_ids),
        "normalizer_version": normalized_round.normalizer_version,
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class _LeaseHeartbeat:
    def __init__(
        self,
        *,
        database_path: str | Path,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        lease_seconds: float,
        interval_seconds: float,
    ) -> None:
        self.database_path = database_path
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_generation = lease_generation
        self.lease_seconds = max(float(lease_seconds), 1)
        self.interval_seconds = max(
            min(float(interval_seconds), self.lease_seconds / 2),
            0.1,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._run,
            name=f"ledgermind-lease-{self.job_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                with SQLiteUnitOfWork(
                    self.database_path, write_transaction=True
                ) as uow:
                    active = uow.raw_rounds.heartbeat(
                        self.job_id,
                        worker_id=self.worker_id,
                        lease_generation=self.lease_generation,
                        lease_seconds=self.lease_seconds,
                    )
                    if active:
                        uow.commit()
            except Exception:
                logger.debug("processing lease heartbeat failed", exc_info=True)
                continue


def _sanitize_candidate(
    value: object,
    normalized_round: NormalizedRound,
) -> HypothesisCandidate:
    if not isinstance(value, HypothesisCandidate):
        raise TypeError("HypothesisGenerator must return HypothesisCandidate values")
    artifacts = tuple(str(redact_value(item)) for item in value.artifacts)
    candidate = HypothesisCandidate(
        title=redact_text(value.title),
        target=redact_text(value.target),
        statement=redact_text(value.statement),
        rationale=redact_text(value.rationale),
        result=redact_text(value.result),
        artifacts=artifacts,
        source_event_ids=value.source_event_ids,
    )
    candidate.validate_source_events(normalized_round)
    return candidate


class RoundProcessingWorker:
    def __init__(
        self,
        *,
        database_path: str | Path,
        generator: HypothesisGenerator,
        worker_id: str | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 30,
        lease_seconds: float = 300,
        heartbeat_interval_seconds: float = 30,
        state: WorkerState | None = None,
    ) -> None:
        self.database_path = database_path
        self.generator = generator
        self.worker_id = worker_id or str(uuid.uuid4())
        self.max_attempts = max(int(max_attempts), 1)
        self.retry_delay_seconds = max(float(retry_delay_seconds), 0)
        self.lease_seconds = max(float(lease_seconds), 1)
        self.heartbeat_interval_seconds = max(float(heartbeat_interval_seconds), 0.1)
        self.state = state or WorkerState("round-processing")
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def create_loop(
        self,
        *,
        poll_interval_seconds: float = 1.0,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
    ) -> GuardedWorkerLoop:
        return GuardedWorkerLoop(
            self,
            state=self.state,
            name="round-processing",
            poll_interval_seconds=poll_interval_seconds,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )

    def run_loop(
        self,
        *,
        poll_interval_seconds: float = 1.0,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.create_loop(
            poll_interval_seconds=poll_interval_seconds,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        ).run()

    def _claim(self) -> tuple[Any, RawRoundRecord] | None:
        if self._stop.is_set():
            return None
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            job = uow.raw_rounds.claim_ready_job(
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if job is None:
                return None
            raw_round = uow.raw_rounds.get(job.raw_round_id)
            if raw_round is None:
                finished = uow.raw_rounds.finish_job(
                    job.job_id,
                    "failed",
                    worker_id=self.worker_id,
                    lease_generation=job.lease_generation,
                    error="raw round not found",
                )
                if not finished:
                    uow.rollback()
                    return (
                        ProcessingResult(job.job_id, job.raw_round_id, "lease_lost"),
                        None,
                    )
                uow.commit()
                return ProcessingResult(job.job_id, job.raw_round_id, "failed"), None
            uow.commit()
            return job, raw_round

    def _error_result(
        self,
        job: Any,
        raw_round: RawRoundRecord,
        *,
        error_code: str,
        error: Exception,
        started_at: str,
    ) -> ProcessingResult:
        detail = redact_text(str(error))[:2_000]
        status = "failed" if job.attempts >= self.max_attempts else "retry_wait"
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            uow.raw_rounds.create_attempt(
                attempt_id=str(uuid.uuid4()),
                raw_round_id=raw_round.raw_round_id,
                job_id=job.job_id,
                pipeline_version=job.pipeline_version,
                normalizer_version=job.normalizer_version,
                provider=getattr(self.generator, "provider", "unknown"),
                model=getattr(self.generator, "model", "unknown"),
                prompt_version=getattr(
                    self.generator, "prompt_version", job.prompt_version
                ),
                schema_version=getattr(self.generator, "schema_version", 1),
                started_at=started_at,
                completed_at=_now(),
                error_code=error_code,
                error_detail=detail,
            )
            finished = uow.raw_rounds.finish_job(
                job.job_id,
                status,
                worker_id=self.worker_id,
                lease_generation=job.lease_generation,
                error=detail,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            if not finished:
                uow.rollback()
                return ProcessingResult(
                    job.job_id, raw_round.raw_round_id, "lease_lost"
                )
            uow.commit()
        return ProcessingResult(job.job_id, raw_round.raw_round_id, status)

    def _persist_success(
        self,
        job: Any,
        raw_round: RawRoundRecord,
        drafts: Sequence[HypothesisCandidate],
        *,
        normalized: NormalizedRound,
        started_at: str,
    ) -> ProcessingResult:
        status = "completed" if drafts else "no_knowledge"
        hypothesis_ids: list[str] = []
        completed_at = _now()
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            attempt_id = str(uuid.uuid4())
            uow.raw_rounds.create_attempt(
                attempt_id=attempt_id,
                raw_round_id=raw_round.raw_round_id,
                job_id=job.job_id,
                pipeline_version=job.pipeline_version,
                normalizer_version=job.normalizer_version,
                provider=getattr(self.generator, "provider", "unknown"),
                model=getattr(self.generator, "model", "unknown"),
                prompt_version=getattr(
                    self.generator, "prompt_version", job.prompt_version
                ),
                schema_version=getattr(self.generator, "schema_version", 1),
                started_at=started_at,
                completed_at=completed_at,
                response_digest=_response_digest(drafts),
            )
            for index, draft in enumerate(drafts):
                hypothesis_id = str(uuid.uuid4())
                hypothesis_ids.append(hypothesis_id)
                content_digest = _digest_candidate(draft)
                idempotency_material = {
                    "memory_space_id": raw_round.memory_space_id,
                    "raw_round_id": raw_round.raw_round_id,
                    "hypothesis_index": index,
                    "content_digest": content_digest,
                    "pipeline_version": job.pipeline_version,
                    "normalizer_version": job.normalizer_version,
                    "prompt_version": job.prompt_version,
                }
                idempotency_key = (
                    "sha256:"
                    + hashlib.sha256(
                        json.dumps(
                            idempotency_material,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                )
                command_payload = {
                    "protocol_version": 1,
                    "command_id": str(uuid.uuid4()),
                    "idempotency_key": idempotency_key,
                    "memory_space_id": raw_round.memory_space_id,
                    "hypothesis": {
                        "hypothesis_id": hypothesis_id,
                        "content_digest": content_digest,
                        "title": draft.title,
                        "target": draft.target,
                        "statement": draft.statement,
                        "rationale": draft.rationale,
                        "result": draft.result,
                        "artifacts": list(draft.artifacts),
                        "evidence": {
                            "source_system": raw_round.source_system,
                            "source_instance_id": raw_round.source_instance_id,
                            "source_profile_id": raw_round.source_profile_id,
                            "source_session_id": raw_round.source_session_id,
                            "source_round_id": raw_round.source_round_id,
                            "raw_round_digest": raw_round.payload_digest,
                            "normalized_round_digest": normalized.normalized_digest,
                            "source_event_ids": list(draft.source_event_ids),
                        },
                        "extraction": {
                            "provider": getattr(self.generator, "provider", "unknown"),
                            "model": getattr(self.generator, "model", "unknown"),
                            "prompt_version": getattr(
                                self.generator,
                                "prompt_version",
                                job.prompt_version,
                            ),
                            "schema_version": getattr(
                                self.generator, "schema_version", 1
                            ),
                            "completed_at": completed_at,
                        },
                    },
                }
                payload_json = json.dumps(
                    command_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                command_id = str(command_payload["command_id"])
                uow.raw_rounds.insert_hypothesis(
                    hypothesis_id=hypothesis_id,
                    raw_round_id=raw_round.raw_round_id,
                    attempt_id=attempt_id,
                    hypothesis_index=index,
                    title=draft.title,
                    target=draft.target,
                    statement=draft.statement,
                    rationale=draft.rationale,
                    result=draft.result,
                    artifacts_json=json.dumps(
                        list(draft.artifacts),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    content_digest=content_digest,
                    status="queued_for_core",
                    core_command_id=command_id,
                )
                uow.raw_rounds.create_core_command(
                    command_id=command_id,
                    command_type="accept_hypothesis",
                    memory_space_id=raw_round.memory_space_id,
                    idempotency_key=idempotency_key,
                    payload_json=payload_json,
                    payload_digest="sha256:"
                    + hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                )
            finished = uow.raw_rounds.finish_job(
                job.job_id,
                status,
                worker_id=self.worker_id,
                lease_generation=job.lease_generation,
            )
            if not finished:
                uow.rollback()
                return ProcessingResult(
                    job.job_id, raw_round.raw_round_id, "lease_lost"
                )
            uow.commit()
        return ProcessingResult(
            job.job_id, raw_round.raw_round_id, status, tuple(hypothesis_ids)
        )

    def _persist_normalized(
        self, raw_round_id: str, normalized: NormalizedRound
    ) -> None:
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            uow.raw_rounds.store_normalized_round(
                raw_round_id=raw_round_id,
                normalizer_version=normalized.normalizer_version,
                payload_json=_normalized_round_payload(normalized),
                payload_digest=normalized.normalized_digest,
            )
            uow.commit()

    def process_once(self) -> ProcessingResult | None:
        claimed = self._claim()
        if claimed is None:
            return None
        if claimed[1] is None:
            return claimed[0]
        job, raw_round = claimed
        started_at = _now()
        try:
            payload = json.loads(raw_round.payload_json)
            normalized = normalize_raw_round(
                payload,
                normalizer_version=job.normalizer_version,
            )
            self._persist_normalized(raw_round.raw_round_id, normalized)
            with _LeaseHeartbeat(
                database_path=self.database_path,
                job_id=job.job_id,
                worker_id=self.worker_id,
                lease_generation=job.lease_generation,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_interval_seconds,
            ):
                drafts = tuple(
                    _sanitize_candidate(item, normalized)
                    for item in self.generator.generate(normalized)
                )
        except Exception as exc:  # noqa: BLE001
            return self._error_result(
                job,
                raw_round,
                error_code="processing_error",
                error=exc,
                started_at=started_at,
            )
        return self._persist_success(
            job,
            raw_round,
            drafts,
            normalized=normalized,
            started_at=started_at,
        )
