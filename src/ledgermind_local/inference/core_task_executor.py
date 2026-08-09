"""Generic Core task executor dispatching only on technical task kinds."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from .cancellation import CancellationToken
from .embedding_provider import EmbeddingBatch, EmbeddingProvider
from .profile_slots import ProfileResolver, ProfileSlot
from .providers.base import ChatMessage, ProviderCancelledError, ProviderTimeoutError
from .structured_json_provider import StructuredJsonProvider

TaskKind = Literal["generate_json", "embed_texts"]
ExecutionStatus = Literal[
    "completed", "failed", "unknown_task_kind", "cancelled", "timeout"
]

T = TypeVar("T")


class ModelRequestSpec(BaseModel):
    """Closed request spec for a generic generate_json task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=32)
    max_output_tokens: int = Field(gt=0, le=50_000)
    response_format: dict[str, object] | None = None


class EmbeddingRequestSpec(BaseModel):
    """Closed request spec for a generic embed_texts task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    texts: tuple[str, ...] = Field(min_length=1)
    purpose: str = Field(min_length=1, max_length=200)
    dimensions: int | None = Field(default=None, gt=0, le=100_000)


class GenericExecutionTask(BaseModel):
    """Technical task contract with no facet, value, or domain semantics.

    ``task_kind`` is left as an open string so the executor can defensively
    report unknown kinds instead of failing at construction time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=200)
    task_kind: str = Field(min_length=1, max_length=200)
    operation: str | None = Field(default=None, max_length=200)
    profile_slot: ProfileSlot
    model_request: ModelRequestSpec | None = None
    embedding_request: EmbeddingRequestSpec | None = None
    expires_at: str | None = Field(default=None, max_length=64)
    lease: dict[str, object] | None = None
    operation_input: dict[str, object] | None = None


class EgressAuditRecord(BaseModel):
    """Content-free egress statistics safe for logging and persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    task_kind: str = Field(min_length=1)
    profile_id: str | None = Field(default=None, max_length=200)
    provider: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=500)
    input_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    status: str = Field(min_length=1)


class GenericExecutionResult(BaseModel):
    """Executor result with typed output and content-free egress audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    task_kind: str = Field(min_length=1)
    operation: str | None = Field(default=None, max_length=200)
    operation_input: dict[str, object] | None = None
    status: ExecutionStatus
    output: dict[str, object] | None = None
    embedding_result: EmbeddingBatch | None = None
    egress_audit: EgressAuditRecord
    error_code: str | None = Field(default=None, max_length=200)


class CoreExecutorError(RuntimeError):
    """Base executor failure with a safe structured code."""

    code = "core_executor_error"


class ExecutionTimeoutError(CoreExecutorError):
    code = "timeout"

    def __init__(self, *args: object) -> None:
        del args
        super().__init__("task execution timed out")


class ExecutionCancelledError(CoreExecutorError):
    code = "cancelled"

    def __init__(self, *args: object) -> None:
        del args
        super().__init__("task execution cancelled")


class CoreTaskExecutor:
    """Dispatch generic tasks to providers by ``task_kind`` only.

    The ``operation`` field is intentionally never inspected: Local does not
    analyze operations. generate_json and embed_texts have separate timeouts,
    and cancellation flows through the shared CancellationToken.
    """

    def __init__(
        self,
        *,
        json_provider: StructuredJsonProvider,
        embedding_provider: EmbeddingProvider,
        profile_resolver: ProfileResolver,
        generate_json_timeout_seconds: float,
        embed_texts_timeout_seconds: float,
    ) -> None:
        self._json_provider = json_provider
        self._embedding_provider = embedding_provider
        self._profile_resolver = profile_resolver
        if generate_json_timeout_seconds <= 0:
            raise ValueError("generate_json_timeout_seconds must be positive")
        if embed_texts_timeout_seconds <= 0:
            raise ValueError("embed_texts_timeout_seconds must be positive")
        self.generate_json_timeout_seconds = float(generate_json_timeout_seconds)
        self.embed_texts_timeout_seconds = float(embed_texts_timeout_seconds)

    def execute(
        self,
        task: GenericExecutionTask,
        cancellation_token: CancellationToken | None = None,
    ) -> GenericExecutionResult:
        if task.task_kind == "generate_json":
            return self._execute_generate_json(task, cancellation_token)
        if task.task_kind == "embed_texts":
            return self._execute_embed_texts(task, cancellation_token)
        return self._unknown_kind_result(task)

    def _execute_generate_json(
        self, task: GenericExecutionTask, token: CancellationToken | None
    ) -> GenericExecutionResult:
        spec = task.model_request
        if spec is None:
            return self._failed_result(
                task, error_code="invalid_request", input_bytes=0
            )
        input_bytes = _model_request_bytes(spec)
        try:
            result = self._run_bounded(
                lambda: self._json_provider.generate_json(
                    memory_space_id=_memory_space_id(task),
                    messages=spec.messages,
                    max_output_tokens=spec.max_output_tokens,
                    response_format=spec.response_format,
                    profile_slot=task.profile_slot,
                    cancellation_token=token,
                ),
                self.generate_json_timeout_seconds,
                token,
            )
        except Exception as exc:  # noqa: BLE001 - classify any provider failure
            return self._failure_result(
                task, exc, input_bytes=input_bytes, profile_id=_resolved_profile_id(exc)
            )
        audit = EgressAuditRecord(
            task_id=task.task_id,
            task_kind=task.task_kind,
            profile_id=result.profile_id,
            provider=result.provider,
            model=result.model,
            input_bytes=input_bytes,
            output_bytes=result.response_bytes,
            status="completed",
        )
        return GenericExecutionResult(
            task_id=task.task_id,
            task_kind=task.task_kind,
            operation=task.operation,
            operation_input=task.operation_input,
            status="completed",
            output=result.data,
            egress_audit=audit,
        )

    def _execute_embed_texts(
        self, task: GenericExecutionTask, token: CancellationToken | None
    ) -> GenericExecutionResult:
        spec = task.embedding_request
        if spec is None:
            return self._failed_result(
                task, error_code="invalid_request", input_bytes=0
            )
        input_bytes = sum(len(text.encode("utf-8")) for text in spec.texts)
        profile = None
        try:
            profile = self._profile_resolver.resolve_profile(
                _memory_space_id(task), task.profile_slot
            )
            batch = self._run_bounded(
                lambda: self._embedding_provider.embed(
                    spec.texts,
                    profile,
                    spec.purpose,
                    cancellation_token=token,
                ),
                self.embed_texts_timeout_seconds,
                token,
            )
        except Exception as exc:  # noqa: BLE001 - classify any provider failure
            return self._failure_result(
                task,
                exc,
                input_bytes=input_bytes,
                profile_id=_resolved_profile_id(exc)
                or (profile.profile_id if profile is not None else None),
            )
        if spec.dimensions is not None and batch.dimensions != spec.dimensions:
            return self._failed_result(
                task,
                error_code="embedding_dimension_mismatch",
                input_bytes=input_bytes,
                profile_id=profile.profile_id,
            )
        audit = EgressAuditRecord(
            task_id=task.task_id,
            task_kind=task.task_kind,
            profile_id=profile.profile_id,
            provider="local",
            model=batch.model,
            input_bytes=input_bytes,
            output_bytes=_embedding_bytes(batch),
            status="completed",
        )
        return GenericExecutionResult(
            task_id=task.task_id,
            task_kind=task.task_kind,
            operation=task.operation,
            operation_input=task.operation_input,
            status="completed",
            embedding_result=batch,
            egress_audit=audit,
        )

    def _run_bounded(
        self,
        work: Callable[[], T],
        timeout_seconds: float,
        token: CancellationToken | None,
    ) -> T:
        if token is None:
            token = CancellationToken()
        token.raise_if_cancelled()

        done = threading.Event()
        outcome: list[tuple[str, object]] = []

        def worker() -> None:
            try:
                outcome.append(("ok", work()))
            except BaseException as exc:  # noqa: BLE001 - relay worker failures
                outcome.append(("error", exc))
            finally:
                done.set()

        thread = threading.Thread(
            target=worker, daemon=True, name="ledgermind-core-executor"
        )
        thread.start()

        deadline = time.monotonic() + timeout_seconds
        cancelled = False
        timed_out = False
        while not done.is_set():
            if token.is_cancelled():
                cancelled = True
                done.wait(timeout=0.5)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                token.cancel()
                done.wait(timeout=0.5)
                break
            time.sleep(0.01)

        if cancelled:
            raise ExecutionCancelledError()
        if timed_out:
            raise ExecutionTimeoutError()
        if outcome:
            kind, value = outcome[0]
            if kind == "ok":
                return cast(T, value)
            raise cast(BaseException, value)
        raise ExecutionCancelledError()

    def _failure_result(
        self,
        task: GenericExecutionTask,
        exc: Exception,
        *,
        input_bytes: int,
        profile_id: str | None = None,
        provider: str | None = None,
    ) -> GenericExecutionResult:
        if isinstance(exc, (ExecutionTimeoutError, ProviderTimeoutError)):
            status: ExecutionStatus = "timeout"
            error_code = "timeout"
        elif isinstance(exc, (ExecutionCancelledError, ProviderCancelledError)):
            status = "cancelled"
            error_code = "cancelled"
        else:
            status = "failed"
            error_code = getattr(exc, "code", "execution_failed")
            if not isinstance(error_code, str) or not error_code:
                error_code = "execution_failed"
        return self._failed_result(
            task,
            error_code=error_code,
            input_bytes=input_bytes,
            profile_id=profile_id,
            provider=provider,
            status=status,
        )

    @staticmethod
    def _failed_result(
        task: GenericExecutionTask,
        *,
        error_code: str,
        input_bytes: int,
        profile_id: str | None = None,
        provider: str | None = None,
        status: ExecutionStatus = "failed",
    ) -> GenericExecutionResult:
        audit = EgressAuditRecord(
            task_id=task.task_id,
            task_kind=task.task_kind,
            profile_id=profile_id,
            provider=provider,
            model=None,
            input_bytes=input_bytes,
            output_bytes=0,
            status=status,
        )
        return GenericExecutionResult(
            task_id=task.task_id,
            task_kind=task.task_kind,
            operation=task.operation,
            operation_input=task.operation_input,
            status=status,
            egress_audit=audit,
            error_code=error_code,
        )

    @staticmethod
    def _unknown_kind_result(task: GenericExecutionTask) -> GenericExecutionResult:
        return CoreTaskExecutor._failed_result(
            task,
            error_code="unknown_task_kind",
            input_bytes=0,
            status="unknown_task_kind",
        )


def _memory_space_id(task: GenericExecutionTask) -> str:
    value = task.lease.get("memory_space_id") if task.lease else None
    return value if isinstance(value, str) else ""


def _model_request_bytes(spec: ModelRequestSpec) -> int:
    payload: dict[str, object] = {
        "messages": [
            {"role": message.role, "content": message.content}
            for message in spec.messages
        ],
        "max_output_tokens": spec.max_output_tokens,
    }
    return len(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _embedding_bytes(batch: EmbeddingBatch) -> int:
    return len(batch.vectors) * batch.dimensions * 4


def _resolved_profile_id(exc: Exception) -> str | None:
    profile_id = getattr(exc, "profile_id", None)
    if isinstance(profile_id, str) and profile_id:
        return profile_id
    return None


__all__ = [
    "CoreExecutorError",
    "CoreTaskExecutor",
    "EgressAuditRecord",
    "EmbeddingRequestSpec",
    "ExecutionCancelledError",
    "ExecutionStatus",
    "ExecutionTimeoutError",
    "GenericExecutionResult",
    "GenericExecutionTask",
    "ModelRequestSpec",
    "TaskKind",
]
