"""Durable delivery worker for Local-to-Core commands."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    DomainRejectedError,
    TransientCoreError,
)
from ledgermind_local.persistence import CoreCommandRecord, SQLiteUnitOfWork
from ledgermind_local.processing.normalizer import redact_text

from .guarded_loop import GuardedWorkerLoop
from .worker_state import WorkerState


@dataclass(frozen=True, slots=True)
class CoreCommandProcessResult:
    command_id: str
    status: str


class CoreCommandWorker:
    """Claim, deliver and finalize Core commands outside SQLite transactions."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        gateway: CoreGateway,
        worker_id: str | None = None,
        max_attempts: int = 5,
        retry_delay_seconds: float = 30,
        lease_seconds: float = 300,
        state: WorkerState | None = None,
    ) -> None:
        self.database_path = database_path
        self.gateway = gateway
        self.worker_id = worker_id or str(uuid.uuid4())
        self.max_attempts = max(int(max_attempts), 1)
        self.retry_delay_seconds = max(float(retry_delay_seconds), 0)
        self.lease_seconds = max(float(lease_seconds), 1)
        self.state = state or WorkerState("core-command")
        self._stop = threading.Event()

    def process_once(self) -> CoreCommandProcessResult | None:
        if self._stop.is_set():
            return None
        command = self._claim()
        if command is None:
            return None
        try:
            if command.command_type != "accept_hypothesis":
                raise DomainRejectedError(
                    "unsupported_command_type",
                    command.command_type,
                )
            try:
                accept_command = AcceptHypothesisCommand.from_json(command.payload_json)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DomainRejectedError(
                    "invalid_command_payload",
                    "Core command payload failed contract validation",
                ) from exc
            result = self.gateway.accept_hypothesis(accept_command)
            if not isinstance(result, AcceptHypothesisResult):
                raise TransientCoreError("CoreGateway returned an invalid result")
            if not result.accepted:
                raise DomainRejectedError(
                    "core_rejected", "Core did not accept hypothesis"
                )
            result_json = result.result_json or json.dumps(
                {
                    "accepted": result.accepted,
                    "duplicate": result.duplicate,
                    "core_reference_id": result.core_reference_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return self._finish(
                command.command_id,
                status="completed",
                result_json=result_json,
                hypothesis_status="accepted_by_core",
            )
        except DomainRejectedError as exc:
            return self._finish(
                command.command_id,
                status="rejected",
                error_code=exc.code,
                error_detail=redact_text(exc.detail)[:2_000],
                hypothesis_status="rejected_by_core",
            )
        except (TransientCoreError, ValueError) as exc:
            return self._finish_retry_or_failure(
                command.command_id,
                attempts=command.attempts,
                error_code="core_delivery_error",
                error=exc,
            )
        except Exception as exc:  # noqa: BLE001
            return self._finish_retry_or_failure(
                command.command_id,
                attempts=command.attempts,
                error_code="core_gateway_error",
                error=exc,
            )

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
            name="core-command",
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

    def _claim(self) -> CoreCommandRecord | None:
        if self._stop.is_set():
            return None
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            command = uow.raw_rounds.claim_core_command(
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if command is None:
                return None
            uow.commit()
            return command

    def _finish_retry_or_failure(
        self,
        command_id: str,
        *,
        attempts: int,
        error_code: str,
        error: Exception,
    ) -> CoreCommandProcessResult:
        status = "failed" if attempts >= self.max_attempts else "retry_wait"
        return self._finish(
            command_id,
            status=status,
            error_code=error_code,
            error_detail=redact_text(str(error))[:2_000],
        )

    def _finish(
        self,
        command_id: str,
        *,
        status: str,
        result_json: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        hypothesis_status: str | None = None,
    ) -> CoreCommandProcessResult:
        with SQLiteUnitOfWork(self.database_path, write_transaction=True) as uow:
            command = uow.raw_rounds.get_core_command(command_id)
            if command is None:
                return CoreCommandProcessResult(command_id, "missing")
            updated = uow.raw_rounds.finish_core_command(
                command_id,
                worker_id=self.worker_id,
                status=status,
                result_json=result_json,
                error_code=error_code,
                error_detail=error_detail,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            if not updated:
                uow.rollback()
                return CoreCommandProcessResult(command_id, "lease_lost")
            if hypothesis_status is not None:
                payload = AcceptHypothesisCommand.from_json(command.payload_json)
                uow.raw_rounds.update_hypothesis_core_status(
                    payload.hypothesis.hypothesis_id,
                    status=hypothesis_status,
                    core_command_id=command_id,
                )
            uow.commit()
        return CoreCommandProcessResult(command_id, status)


__all__ = ["CoreCommandProcessResult", "CoreCommandWorker"]
