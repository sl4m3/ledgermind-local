"""Local worker for generic Core execution tasks."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.contracts import (
    CoreExecutionTask,
    DomainRejectedError,
    FailExecutionTaskCommand,
    PollExecutionTasksCommand,
    SubmitExecutionResultCommand,
    TransientCoreError,
)
from ledgermind_local.inference.core_task_executor import (
    CoreTaskExecutor,
    EmbeddingRequestSpec,
    GenericExecutionTask,
    ModelRequestSpec,
)
from ledgermind_local.inference.profile_slots import ProfileSlot
from ledgermind_local.inference.providers.base import (
    ChatMessage,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTransportError,
    TransientProviderError,
)
from ledgermind_local.inference.secrets import SecretNotFoundError
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations

logger = logging.getLogger(__name__)

_SAFE_ERROR_CODES = frozenset(
    {
        "execution_error",
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
        "core_rejected_stale_execution_task",
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


def _safe_error_code(value: str | None, fallback: str = "execution_error") -> str:
    if value in _SAFE_ERROR_CODES:
        return value
    return fallback


class ExecutionFailureClassification:
    """Safe error code and retry policy sent back to Core."""

    def __init__(
        self, error_code: str, retryable: bool, retry_after_seconds: int = 60
    ) -> None:
        if not error_code.strip():
            raise ValueError("error_code must not be empty")
        if not 0 <= retry_after_seconds <= 86_400:
            raise ValueError("retry_after_seconds must be between 0 and 86400")
        if not retryable and retry_after_seconds != 0:
            raise ValueError("permanent failures must not request a retry delay")
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


def classify_execution_error(exc: BaseException) -> ExecutionFailureClassification:
    """Classify provider/Core failures without serializing exception details."""

    if isinstance(exc, (ProviderTimeoutError, TimeoutError)):
        return ExecutionFailureClassification("provider_timeout", True)
    if isinstance(exc, (ProviderTransportError, ConnectionError, OSError)):
        return ExecutionFailureClassification("provider_transport_error", True)
    if isinstance(exc, TransientProviderError):
        return ExecutionFailureClassification("provider_unavailable", True)
    error_type = type(exc).__name__
    if error_type == "InferenceProfileNotFoundError":
        return ExecutionFailureClassification("profile_not_found", False, 0)
    if error_type == "InferenceProfileDisabledError":
        return ExecutionFailureClassification("profile_disabled", False, 0)
    if error_type == "InferenceInputTooLargeError":
        return ExecutionFailureClassification("input_too_large", False, 0)
    if error_type == "InferenceResponseValidationError":
        return ExecutionFailureClassification("invalid_model_output", False, 0)
    if isinstance(exc, (ProviderAuthenticationError, ProviderConfigurationError)):
        return ExecutionFailureClassification("provider_configuration_error", False, 0)
    if isinstance(exc, SecretNotFoundError):
        return ExecutionFailureClassification("provider_secret_missing", False, 0)
    if isinstance(exc, (ProviderResponseError, ValueError, TypeError)):
        return ExecutionFailureClassification("invalid_model_output", False, 0)
    if isinstance(exc, TransientCoreError):
        return ExecutionFailureClassification("core_unavailable", True)
    if isinstance(exc, DomainRejectedError):
        code = re.sub(r"[^a-z0-9]+", "_", exc.code.lower()).strip("_")
        return ExecutionFailureClassification(
            _safe_error_code(
                f"core_rejected_{code or 'command'}",
                "core_rejected_command",
            ),
            False,
            0,
        )
    if isinstance(exc, ProviderError):
        return ExecutionFailureClassification("provider_error", False, 0)

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code <= 599):
        return ExecutionFailureClassification("provider_unavailable", True)
    message = str(exc).lower()
    if any(marker in message for marker in ("429", "temporar", "unavailable", "connection")):
        return ExecutionFailureClassification("provider_unavailable", True)
    if "timeout" in message or "timed out" in message:
        return ExecutionFailureClassification("provider_timeout", True)
    if isinstance(exc, RuntimeError):
        return ExecutionFailureClassification("provider_error", False, 0)
    return ExecutionFailureClassification("execution_error", False, 0)


ConnectionFactory = Callable[[str | Path], sqlite3.Connection]


class CoreExecutionTaskWorker:
    """Execute generic Core tasks without interpreting operation metadata."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        gateway: CoreGateway,
        executor: CoreTaskExecutor,
        worker_id: str,
        poll_limit: int = 10,
        lease_seconds: int = 300,
        connection_factory: ConnectionFactory = open_sqlite_connection,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not 1 <= poll_limit <= 100:
            raise ValueError("poll_limit must be between 1 and 100")
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        self._database_path = str(database_path)
        self._gateway = gateway
        self._executor = executor
        self._worker_id = worker_id
        self._poll_limit = poll_limit
        self._lease_seconds = lease_seconds
        self._connection_factory = connection_factory
        self._closed = False
        require_capabilities = getattr(gateway, "require_capabilities", None)
        if callable(require_capabilities):
            require_capabilities("execution_tasks")

    def process_once(self) -> int:
        if self._closed:
            raise RuntimeError("Core execution task worker is closed")
        connection = self._connection_factory(self._database_path)
        processed = 0
        try:
            migrations.apply_migrations(connection)
            spaces = connection.execute(
                "SELECT memory_space_id FROM memory_spaces ORDER BY memory_space_id"
            ).fetchall()
            for row in spaces:
                memory_space_id = str(row[0])
                try:
                    polled = self._gateway.poll_execution_tasks(
                        PollExecutionTasksCommand(
                            request_id=self._request_id("poll"),
                            memory_space_id=memory_space_id,
                            worker_id=self._worker_id,
                            limit=self._poll_limit,
                            lease_seconds=self._lease_seconds,
                        )
                    )
                except Exception:  # noqa: BLE001 - isolate one space's backlog
                    logger.warning(
                        "Core execution task poll failed",
                        extra={
                            "worker": self._worker_id,
                            "memory_space_id": memory_space_id,
                            "error_code": "core_poll_error",
                        },
                    )
                    continue
                for raw_task in polled.tasks:
                    processed += 1
                    self._process_task(raw_task, memory_space_id)
        finally:
            connection.close()
        return processed

    def close(self) -> None:
        self._closed = True

    def _process_task(self, raw_task: dict[str, object], memory_space_id: str) -> None:
        task_id = str(raw_task.get("task_id", "unknown"))
        try:
            task = _local_execution_task(raw_task, memory_space_id)
            result = self._executor.execute(task)
            if result.status == "completed":
                submitted = self._gateway.submit_execution_result(
                    SubmitExecutionResultCommand(
                        request_id=self._request_id("submit"),
                        task_id=task.task_id,
                        memory_space_id=memory_space_id,
                        worker_id=self._worker_id,
                        result=result.model_dump(mode="json"),
                    )
                )
                if not submitted.accepted:
                    raise TransientCoreError("Core did not accept execution result")
                return
            self._gateway.fail_execution_task(
                FailExecutionTaskCommand(
                    request_id=self._request_id("fail"),
                    task_id=task.task_id,
                    memory_space_id=memory_space_id,
                    worker_id=self._worker_id,
                    error_code=result.error_code or result.status,
                    retryable=result.status in {"timeout", "cancelled"},
                    retry_after_seconds=60
                    if result.status in {"timeout", "cancelled"}
                    else 0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - release every leased task
            classification = classify_execution_error(exc)
            logger.warning(
                "Core execution task failed",
                extra={
                    "worker": self._worker_id,
                    "task_id": task_id,
                    "error_code": classification.error_code,
                },
            )
            self._gateway.fail_execution_task(
                FailExecutionTaskCommand(
                    request_id=self._request_id("fail"),
                    task_id=task_id,
                    memory_space_id=memory_space_id,
                    worker_id=self._worker_id,
                    error_code=classification.error_code,
                    retryable=classification.retryable,
                    retry_after_seconds=classification.retry_after_seconds,
                )
            )

    def _request_id(self, operation: str) -> str:
        return f"{self._worker_id}:{operation}:{uuid.uuid4()}"


def _local_execution_task(
    raw_task: dict[str, object], memory_space_id: str
) -> GenericExecutionTask:
    """Convert the language-neutral wire task to the Local executor model."""

    wire = CoreExecutionTask.from_payload(raw_task)
    if wire.memory_space_id != memory_space_id:
        raise ValueError("execution task memory space does not match poll scope")
    model_request = None
    if wire.model_request is not None:
        model_request = ModelRequestSpec(
            messages=tuple(
                ChatMessage.model_validate(message)
                for message in wire.model_request.get("messages", [])
            ),
            max_output_tokens=int(wire.model_request["max_output_tokens"]),
            response_format={"type": wire.model_request["response_format"]}
            if wire.model_request.get("response_format") == "json_object"
            else None,
        )
    embedding_request = None
    if wire.embedding_request is not None:
        embedding_request = EmbeddingRequestSpec(
            texts=tuple(str(text) for text in wire.embedding_request.get("texts", [])),
            purpose=str(wire.embedding_request["purpose"]),
            dimensions=(
                int(wire.embedding_request["dimensions"])
                if wire.embedding_request.get("dimensions") is not None
                else None
            ),
        )
    return GenericExecutionTask(
        task_id=wire.task_id,
        task_kind=wire.task_kind,
        operation=wire.operation,
        profile_slot=ProfileSlot(wire.profile_slot),
        model_request=model_request,
        embedding_request=embedding_request,
        expires_at=wire.expires_at,
        lease={"memory_space_id": memory_space_id, "value": wire.lease},
        operation_input=wire.operation_input,
    )


__all__ = ["CoreExecutionTaskWorker", "ExecutionFailureClassification", "classify_execution_error"]
