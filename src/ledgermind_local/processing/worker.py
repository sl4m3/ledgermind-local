"""Crash-tolerant RawRound processing worker."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..persistence import RawRoundRecord, SQLiteUnitOfWork
from .generator import HypothesisDraft, HypothesisGenerator
from .models import NormalizedRound
from .normalizer import normalize_raw_round, redact_text, redact_value


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    job_id: str
    raw_round_id: str
    status: str
    hypothesis_ids: tuple[str, ...] = ()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest_draft(draft: HypothesisDraft) -> str:
    material = {
        "title": draft.title,
        "target": draft.target,
        "statement": draft.statement,
        "rationale": draft.rationale,
        "result": draft.result,
        "artifacts": list(draft.artifacts),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _response_digest(drafts: Sequence[HypothesisDraft]) -> str:
    material = [
        {
            "title": draft.title,
            "target": draft.target,
            "statement": draft.statement,
            "rationale": draft.rationale,
            "result": draft.result,
            "artifacts": list(draft.artifacts),
        }
        for draft in drafts
    ]
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sanitize_draft(value: object) -> HypothesisDraft:
    if not isinstance(value, HypothesisDraft):
        raise TypeError("HypothesisGenerator must return HypothesisDraft values")
    artifacts = tuple(str(redact_value(item)) for item in value.artifacts)
    return HypothesisDraft(
        title=redact_text(value.title),
        target=redact_text(value.target),
        statement=redact_text(value.statement),
        rationale=redact_text(value.rationale),
        result=redact_text(value.result),
        artifacts=artifacts,
    )


class RoundProcessingWorker:
    def __init__(
        self,
        *,
        database_path: str | Path,
        generator: HypothesisGenerator,
        bridge: Callable[[RawRoundRecord, HypothesisDraft], object] | None = None,
        worker_id: str | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 30,
    ) -> None:
        self.database_path = database_path
        self.generator = generator
        self.bridge = bridge
        self.worker_id = worker_id or str(uuid.uuid4())
        self.max_attempts = max(int(max_attempts), 1)
        self.retry_delay_seconds = max(float(retry_delay_seconds), 0)

    def _claim(self) -> tuple[Any, RawRoundRecord] | None:
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            job = uow.raw_rounds.claim_ready_job(worker_id=self.worker_id)
            if job is None:
                return None
            raw_round = uow.raw_rounds.get(job.raw_round_id)
            if raw_round is None:
                uow.raw_rounds.finish_job(job.job_id, "failed", error="raw round not found")
                uow.commit()
                return ProcessingResult(job.job_id, job.raw_round_id, "failed"), None  # type: ignore[return-value]
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
                prompt_version=getattr(self.generator, "prompt_version", job.prompt_version),
                schema_version=getattr(self.generator, "schema_version", 1),
                started_at=started_at,
                completed_at=_now(),
                error_code=error_code,
                error_detail=detail,
            )
            uow.raw_rounds.finish_job(
                job.job_id,
                status,
                error=detail,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            uow.commit()
        return ProcessingResult(job.job_id, raw_round.raw_round_id, status)

    def _persist_success(
        self,
        job: Any,
        raw_round: RawRoundRecord,
        drafts: Sequence[HypothesisDraft],
        *,
        started_at: str,
    ) -> ProcessingResult:
        status = "completed" if drafts else "no_knowledge"
        hypothesis_ids: list[str] = []
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
                prompt_version=getattr(self.generator, "prompt_version", job.prompt_version),
                schema_version=getattr(self.generator, "schema_version", 1),
                started_at=started_at,
                completed_at=_now(),
                response_digest=_response_digest(drafts),
            )
            for index, draft in enumerate(drafts):
                hypothesis_id = str(uuid.uuid4())
                hypothesis_ids.append(hypothesis_id)
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
                    artifacts_json=json.dumps(list(draft.artifacts), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    content_digest=_digest_draft(draft),
                )
            uow.raw_rounds.finish_job(job.job_id, status)
            uow.commit()
        return ProcessingResult(job.job_id, raw_round.raw_round_id, status, tuple(hypothesis_ids))

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
            normalized: NormalizedRound = normalize_raw_round(payload)
            drafts = tuple(_sanitize_draft(item) for item in self.generator.generate(normalized))
            if self.bridge is not None:
                for draft in drafts:
                    self.bridge(raw_round, draft)
        except Exception as exc:  # noqa: BLE001
            return self._error_result(
                job,
                raw_round,
                error_code="processing_error",
                error=exc,
                started_at=started_at,
            )
        return self._persist_success(job, raw_round, drafts, started_at=started_at)
