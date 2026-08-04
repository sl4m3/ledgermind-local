"""Local worker for leased Core generative merge tasks."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.contracts import (
    DomainRejectedError,
    TransientCoreError,
)
from ledgermind_local.core_gateway.model_task_contracts import (
    CoreModelTask,
    FailModelTaskCommand,
    FailModelTaskResult,
    PollModelTasksCommand,
    SubmitModelResultCommand,
)
from ledgermind_local.inference.providers.base import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTransportError,
    TransientProviderError,
)
from ledgermind_local.inference.schemas import MergeProposal
from ledgermind_local.inference.secrets import SecretNotFoundError
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations

logger = logging.getLogger(__name__)

_SAFE_ERROR_CODES = frozenset(
    {
        "model_task_error",
        "provider_timeout",
        "provider_transport_error",
        "provider_unavailable",
        "provider_configuration_error",
        "provider_secret_missing",
        "provider_error",
        "invalid_model_output",
        "profile_not_found",
        "profile_disabled",
        "input_too_large",
        "profile_missing",
        "core_unavailable",
        "core_poll_error",
        "core_delivery_error",
        "core_rejected_command",
        "core_rejected_stale_model_task",
        "core_rejected_version_conflict",
        "core_rejected_invalid_request",
        "core_rejected_not_found",
        "core_rejected_idempotency_conflict",
        "core_rejected_integrity_violation",
        "retry_exhausted",
        "expired",
        "permanent_failure",
    }
)


def _safe_error_code(value: str | None, fallback: str = "model_task_error") -> str:
    if value in _SAFE_ERROR_CODES:
        return value
    return fallback


class MergeProposalBroker(Protocol):
    def generate_merge_proposal(
        self,
        *,
        memory_space_id: str,
        model_input: dict[str, object],
        profile_id: str,
    ) -> MergeProposal: ...


@dataclass(frozen=True, slots=True)
class ModelTaskFailureClassification:
    """Safe error code and retry policy sent back to Core."""

    error_code: str
    retryable: bool
    retry_after_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.error_code.strip():
            raise ValueError("error_code must not be empty")
        if not 0 <= self.retry_after_seconds <= 86_400:
            raise ValueError("retry_after_seconds must be between 0 and 86400")
        if not self.retryable and self.retry_after_seconds != 0:
            raise ValueError("permanent failures must not request a retry delay")


@dataclass(frozen=True, slots=True)
class CoreModelTaskWorkerStats:
    fetched: int = 0
    completed: int = 0
    duplicates: int = 0
    failed: int = 0
    released: int = 0
    retryable_failures: int = 0
    permanent_failures: int = 0
    retry_scheduled: int = 0
    terminal_failures: int = 0
    provider_failures: int = 0
    core_poll_failures: int = 0
    core_delivery_failures: int = 0
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "fetched",
            "completed",
            "duplicates",
            "failed",
            "released",
            "retryable_failures",
            "permanent_failures",
            "retry_scheduled",
            "terminal_failures",
            "provider_failures",
            "core_poll_failures",
            "core_delivery_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.last_error_code is not None and self.last_error_code not in _SAFE_ERROR_CODES:
            raise ValueError("last_error_code must be a safe error code")

    @property
    def degraded(self) -> bool:
        """Whether this iteration observed a failure that health should expose."""

        return any(
            value > 0
            for value in (
                self.failed,
                self.retryable_failures,
                self.permanent_failures,
                self.retry_scheduled,
                self.terminal_failures,
                self.provider_failures,
                self.core_poll_failures,
                self.core_delivery_failures,
            )
        )

    @property
    def made_progress(self) -> bool:
        """Whether Local completed work or changed a Core task state."""

        return any(
            value > 0
            for value in (
                self.completed,
                self.released,
                self.retry_scheduled,
                self.terminal_failures,
            )
        )


ConnectionFactory = Callable[[str | Path], sqlite3.Connection]


def classify_model_task_error(exc: BaseException) -> ModelTaskFailureClassification:
    """Classify provider/Core failures without serializing exception details."""

    if isinstance(exc, (ProviderTimeoutError, TimeoutError)):
        return ModelTaskFailureClassification("provider_timeout", True)
    if isinstance(exc, (ProviderTransportError, ConnectionError, OSError)):
        return ModelTaskFailureClassification("provider_transport_error", True)
    if isinstance(exc, TransientProviderError):
        return ModelTaskFailureClassification("provider_unavailable", True)
    error_type = type(exc).__name__
    if error_type == "InferenceProfileNotFoundError":
        return ModelTaskFailureClassification("profile_not_found", False, 0)
    if error_type == "InferenceProfileDisabledError":
        return ModelTaskFailureClassification("profile_disabled", False, 0)
    if error_type == "InferenceInputTooLargeError":
        return ModelTaskFailureClassification("input_too_large", False, 0)
    if error_type == "InferenceResponseValidationError":
        return ModelTaskFailureClassification("invalid_model_output", False, 0)
    if isinstance(
        exc, (ProviderAuthenticationError, ProviderConfigurationError)
    ):
        return ModelTaskFailureClassification("provider_configuration_error", False, 0)
    if isinstance(exc, SecretNotFoundError):
        return ModelTaskFailureClassification("provider_secret_missing", False, 0)
    if isinstance(exc, (ProviderResponseError, ValueError, TypeError)):
        return ModelTaskFailureClassification("invalid_model_output", False, 0)
    if isinstance(exc, TransientCoreError):
        return ModelTaskFailureClassification("core_unavailable", True)
    if isinstance(exc, DomainRejectedError):
        code = re.sub(r"[^a-z0-9]+", "_", exc.code.lower()).strip("_")
        return ModelTaskFailureClassification(
            _safe_error_code(
                f"core_rejected_{code or 'command'}",
                "core_rejected_command",
            ),
            False,
            0,
        )
    if isinstance(exc, ProviderError):
        return ModelTaskFailureClassification("provider_error", False, 0)

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code <= 599):
        return ModelTaskFailureClassification("provider_unavailable", True)
    message = str(exc).lower()
    if any(marker in message for marker in ("429", "temporar", "unavailable", "connection")):
        return ModelTaskFailureClassification("provider_unavailable", True)
    if "timeout" in message or "timed out" in message:
        return ModelTaskFailureClassification("provider_timeout", True)
    if isinstance(exc, RuntimeError):
        return ModelTaskFailureClassification("provider_error", False, 0)
    return ModelTaskFailureClassification("model_task_error", False, 0)


class CoreModelTaskWorker:
    """Poll Core leases and execute merge proposals through Local InferenceBroker."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        gateway: CoreGateway,
        broker: MergeProposalBroker,
        worker_id: str,
        poll_limit: int = 10,
        lease_seconds: int = 300,
        retry_after_seconds: int = 60,
        connection_factory: ConnectionFactory = open_sqlite_connection,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not 1 <= poll_limit <= 100:
            raise ValueError("poll_limit must be between 1 and 100")
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        if not 0 <= retry_after_seconds <= 86_400:
            raise ValueError("retry_after_seconds must be between 0 and 86400")
        self._database_path = str(database_path)
        self._gateway = gateway
        self._broker = broker
        self._worker_id = worker_id
        self._poll_limit = poll_limit
        self._lease_seconds = lease_seconds
        self._retry_after_seconds = retry_after_seconds
        self._connection_factory = connection_factory
        self._closed = False
        require_capabilities = getattr(gateway, "require_capabilities", None)
        if callable(require_capabilities):
            require_capabilities("model_tasks")

    def process_once(self) -> CoreModelTaskWorkerStats:
        if self._closed:
            raise RuntimeError("Core model task worker is closed")
        stats = CoreModelTaskWorkerStats()
        connection = self._connection_factory(self._database_path)
        try:
            migrations.apply_migrations(connection)
            rows = connection.execute(
                """
                SELECT memory_space_id
                FROM memory_spaces
                ORDER BY memory_space_id ASC
                """
            ).fetchall()
            for row in rows:
                memory_space_id = str(row[0])
                binding = connection.execute(
                    """
                    SELECT merge_profile_id
                    FROM memory_space_inference_profiles
                    WHERE memory_space_id = ?
                    """,
                    (memory_space_id,),
                ).fetchone()
                merge_profile_id = binding[0] if binding is not None else None
                try:
                    polled = self._gateway.poll_model_tasks(
                        PollModelTasksCommand(
                            request_id=self._request_id("poll"),
                            memory_space_id=memory_space_id,
                            worker_id=self._worker_id,
                            limit=self._poll_limit,
                            lease_seconds=self._lease_seconds,
                        )
                    )
                except Exception:  # noqa: BLE001 - poll failures are isolated per memory space
                    # Polling never touches provider failure counters: no task was
                    # handed to Local and there is no lease to report back.
                    logger.warning(
                        "Core model task poll failed",
                        extra={
                            "worker": self._worker_id,
                            "memory_space_id": memory_space_id,
                            "error_code": "core_poll_error",
                        },
                    )
                    stats = _add_stats(
                        stats,
                        failed=1,
                        core_poll_failures=1,
                        last_error_code="core_poll_error",
                    )
                    continue
                stats = _add_stats(stats, fetched=len(polled.tasks))
                for task in polled.tasks:
                    if merge_profile_id is None:
                        stats = _release_task(
                            stats,
                            task,
                            self._gateway,
                            self._worker_id,
                            ModelTaskFailureClassification("profile_missing", False, 0),
                            self._request_id,
                        )
                        continue
                    try:
                        proposal = self._broker.generate_merge_proposal(
                            memory_space_id=task.memory_space_id,
                            model_input=task.model_input,
                            profile_id=merge_profile_id,
                        )
                    except Exception as exc:  # noqa: BLE001 - provider transports vary
                        stats = _release_task(
                            stats,
                            task,
                            self._gateway,
                            self._worker_id,
                            _worker_classification(exc, self._retry_after_seconds),
                            self._request_id,
                            provider_failure=True,
                        )
                        continue
                    try:
                        submitted = self._gateway.submit_model_result(
                            SubmitModelResultCommand(
                                request_id=self._request_id("submit"),
                                task_id=task.task_id,
                                memory_space_id=task.memory_space_id,
                                worker_id=self._worker_id,
                                result=proposal.model_dump(mode="json"),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - delivery must not orphan lease
                        stats = _release_task(
                            stats,
                            task,
                            self._gateway,
                            self._worker_id,
                            _core_worker_classification(
                                exc, self._retry_after_seconds
                            ),
                            self._request_id,
                            core_delivery_failure=True,
                        )
                        continue
                    if not submitted.accepted:
                        stats = _release_task(
                            stats,
                            task,
                            self._gateway,
                            self._worker_id,
                            ModelTaskFailureClassification(
                                "core_delivery_error",
                                True,
                                self._retry_after_seconds,
                            ),
                            self._request_id,
                            core_delivery_failure=True,
                        )
                        continue
                    stats = _add_stats(
                        stats,
                        completed=1,
                        duplicates=1 if submitted.duplicate else 0,
                    )
        finally:
            connection.close()
        return stats

    def close(self) -> None:
        self._closed = True

    def _request_id(self, operation: str) -> str:
        return f"{self._worker_id}:{operation}:{uuid.uuid4()}"


def _worker_classification(
    exc: BaseException, retry_after_seconds: int
) -> ModelTaskFailureClassification:
    classification = classify_model_task_error(exc)
    if not classification.retryable:
        return classification
    return ModelTaskFailureClassification(
        classification.error_code,
        True,
        retry_after_seconds,
    )


def _core_worker_classification(
    exc: BaseException, retry_after_seconds: int
) -> ModelTaskFailureClassification:
    """Classify a Core delivery error without treating it as a provider failure."""

    if isinstance(exc, DomainRejectedError):
        classification = classify_model_task_error(exc)
        if not classification.retryable:
            return classification
    return ModelTaskFailureClassification(
        "core_delivery_error",
        True,
        retry_after_seconds,
    )


def _release_task(
    stats: CoreModelTaskWorkerStats,
    task: CoreModelTask,
    gateway: CoreGateway,
    worker_id: str,
    classification: ModelTaskFailureClassification,
    request_id_factory: Callable[[str], str],
    *,
    provider_failure: bool = False,
    core_delivery_failure: bool = False,
) -> CoreModelTaskWorkerStats:
    try:
        result = gateway.fail_model_task(
            FailModelTaskCommand(
                request_id=request_id_factory("fail"),
                task_id=task.task_id,
                memory_space_id=task.memory_space_id,
                worker_id=worker_id,
                error_code=classification.error_code,
                retryable=classification.retryable,
                retry_after_seconds=classification.retry_after_seconds,
                failed_at=datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            )
        )
        if not isinstance(result, FailModelTaskResult):
            raise TypeError("Core returned an invalid model task failure result")
    except Exception as exc:  # noqa: BLE001 - leave stats honest if release also fails
        logger.warning(
            "Core model task release failed",
            extra={
                "worker": worker_id,
                "task_id": task.task_id,
                "error_code": "core_delivery_error",
                "error_type": type(exc).__name__,
            },
        )
        return _add_stats(
            stats,
            failed=1,
            retryable_failures=1 if classification.retryable else 0,
            permanent_failures=0 if classification.retryable else 1,
            provider_failures=1 if provider_failure else 0,
            core_delivery_failures=1 + (1 if core_delivery_failure else 0),
            last_error_code="core_delivery_error",
        )

    return _add_stats(
        stats,
        failed=1,
        released=1,
        retryable_failures=1 if classification.retryable else 0,
        permanent_failures=0 if classification.retryable else 1,
        retry_scheduled=1 if result.retry_scheduled else 0,
        terminal_failures=1 if result.terminal else 0,
        provider_failures=1 if provider_failure else 0,
        core_delivery_failures=1 if core_delivery_failure else 0,
        last_error_code=_safe_error_code(
            result.last_error_code,
            classification.error_code,
        ),
    )


def _add_stats(
    stats: CoreModelTaskWorkerStats,
    *,
    fetched: int = 0,
    completed: int = 0,
    duplicates: int = 0,
    failed: int = 0,
    released: int = 0,
    retryable_failures: int = 0,
    permanent_failures: int = 0,
    retry_scheduled: int = 0,
    terminal_failures: int = 0,
    provider_failures: int = 0,
    core_poll_failures: int = 0,
    core_delivery_failures: int = 0,
    last_error_code: str | None = None,
) -> CoreModelTaskWorkerStats:
    return CoreModelTaskWorkerStats(
        fetched=stats.fetched + fetched,
        completed=stats.completed + completed,
        duplicates=stats.duplicates + duplicates,
        failed=stats.failed + failed,
        released=stats.released + released,
        retryable_failures=stats.retryable_failures + retryable_failures,
        permanent_failures=stats.permanent_failures + permanent_failures,
        retry_scheduled=stats.retry_scheduled + retry_scheduled,
        terminal_failures=stats.terminal_failures + terminal_failures,
        provider_failures=stats.provider_failures + provider_failures,
        core_poll_failures=stats.core_poll_failures + core_poll_failures,
        core_delivery_failures=stats.core_delivery_failures + core_delivery_failures,
        last_error_code=(
            stats.last_error_code
            if last_error_code is None
            else _safe_error_code(last_error_code)
        ),
    )


__all__ = [
    "CoreModelTaskWorker",
    "CoreModelTaskWorkerStats",
    "ModelTaskFailureClassification",
    "classify_model_task_error",
]
