"""Strict Local contracts for Core model-task polling and result submission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _timestamp(value: object, name: str) -> str:
    text = _required_text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return text


def _strict_mapping(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class CoreModelTask:
    task_id: str
    operation: str
    memory_space_id: str
    expected_versions: dict[str, int]
    expires_at: str
    model_input: dict[str, Any]
    lease_expires_at: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.task_id, "task_id")
        _required_text(self.memory_space_id, "memory_space_id")
        if self.operation != "merge_knowledge":
            raise ValueError("unsupported model task operation")
        if len(self.expected_versions) < 2:
            raise ValueError("merge task requires at least two expected versions")
        for reference, version in self.expected_versions.items():
            _required_text(reference, "expected version reference")
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise ValueError("expected versions must be non-negative integers")
        _timestamp(self.expires_at, "expires_at")
        if not isinstance(self.model_input, dict):
            raise TypeError("model_input must be an object")
        if self.lease_expires_at is not None:
            _timestamp(self.lease_expires_at, "lease_expires_at")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> CoreModelTask:
        _strict_mapping(
            payload,
            {
                "task_id",
                "operation",
                "memory_space_id",
                "expected_versions",
                "expires_at",
                "model_input",
                "lease_expires_at",
            },
            "model task",
        )
        expected_versions = payload["expected_versions"]
        if not isinstance(expected_versions, dict):
            raise TypeError("expected_versions must be an object")
        model_input = payload["model_input"]
        if not isinstance(model_input, dict):
            raise TypeError("model_input must be an object")
        task_id = _required_text(payload.get("task_id"), "task_id")
        operation = _required_text(payload.get("operation"), "operation")
        memory_space_id = _required_text(
            payload.get("memory_space_id"), "memory_space_id"
        )
        expires_at = _required_text(payload.get("expires_at"), "expires_at")
        normalized_versions: dict[str, int] = {}
        for reference, version in expected_versions.items():
            normalized_reference = _required_text(reference, "expected version reference")
            normalized_versions[normalized_reference] = version
        return cls(
            task_id=task_id,
            operation=operation,
            memory_space_id=memory_space_id,
            expected_versions=normalized_versions,
            expires_at=expires_at,
            model_input=dict(model_input),
            lease_expires_at=(
                _required_text(payload["lease_expires_at"], "lease_expires_at")
                if payload.get("lease_expires_at") is not None
                else None
            ),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "task_id": self.task_id,
            "operation": self.operation,
            "memory_space_id": self.memory_space_id,
            "expected_versions": dict(self.expected_versions),
            "expires_at": self.expires_at,
            "model_input": dict(self.model_input),
        }
        if self.lease_expires_at is not None:
            payload["lease_expires_at"] = self.lease_expires_at
        return payload


@dataclass(frozen=True, slots=True)
class PollModelTasksCommand:
    request_id: str
    memory_space_id: str
    worker_id: str
    limit: int = 10
    lease_seconds: int = 60

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.memory_space_id, "memory_space_id")
        _required_text(self.worker_id, "worker_id")
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 1 <= self.lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")

    def to_payload(self) -> dict[str, object]:
        return {
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "limit": self.limit,
            "lease_seconds": self.lease_seconds,
        }


@dataclass(frozen=True, slots=True)
class PollModelTasksResult:
    tasks: tuple[CoreModelTask, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class SubmitModelResultCommand:
    request_id: str
    task_id: str
    memory_space_id: str
    worker_id: str
    result: dict[str, Any]

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.memory_space_id, "memory_space_id")
        _required_text(self.worker_id, "worker_id")
        if not isinstance(self.result, dict):
            raise TypeError("result must be an object")

    def to_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "result": dict(self.result),
        }


@dataclass(frozen=True, slots=True)
class SubmitModelResult:
    accepted: bool
    duplicate: bool
    status: str


@dataclass(frozen=True, slots=True)
class FailModelTaskCommand:
    """Classified failure reported for a leased Core model task."""

    request_id: str
    task_id: str
    memory_space_id: str
    worker_id: str
    error_code: str
    retryable: bool
    retry_after_seconds: int
    failed_at: str

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.memory_space_id, "memory_space_id")
        _required_text(self.worker_id, "worker_id")
        _required_text(self.error_code, "error_code")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if (
            isinstance(self.retry_after_seconds, bool)
            or not isinstance(self.retry_after_seconds, int)
            or not 0 <= self.retry_after_seconds <= 86_400
        ):
            raise ValueError("retry_after_seconds must be between 0 and 86400")
        _timestamp(self.failed_at, "failed_at")

    def to_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "failed_at": self.failed_at,
        }


@dataclass(frozen=True, slots=True)
class FailModelTaskResult:
    """Result of atomically transitioning a failed model task."""

    status: str
    attempts: int
    available_at: str | None
    last_error_code: str | None
    failed_at: str | None
    completed_at: str | None

    @property
    def retry_scheduled(self) -> bool:
        """Core accepted the failure and scheduled another claim."""

        return self.status == "pending"

    @property
    def terminal(self) -> bool:
        """Core made the failure terminal for this task."""

        return self.status == "failed"

    def __post_init__(self) -> None:
        if self.status not in {"pending", "failed"}:
            raise ValueError("model task failure status is invalid")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise TypeError("attempts must be an integer")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        for value, name in (
            (self.available_at, "available_at"),
            (self.failed_at, "failed_at"),
            (self.completed_at, "completed_at"),
        ):
            if value is not None:
                _timestamp(value, name)
        if self.last_error_code is not None:
            _required_text(self.last_error_code, "last_error_code")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FailModelTaskResult:
        _strict_mapping(
            payload,
            {
                "status",
                "attempts",
                "available_at",
                "last_error_code",
                "failed_at",
                "completed_at",
            },
            "fail model task result",
        )
        return cls(
            status=payload["status"],
            attempts=payload["attempts"],
            available_at=payload.get("available_at"),
            last_error_code=payload.get("last_error_code"),
            failed_at=payload.get("failed_at"),
            completed_at=payload.get("completed_at"),
        )


__all__ = [
    "CoreModelTask",
    "FailModelTaskCommand",
    "FailModelTaskResult",
    "PollModelTasksCommand",
    "PollModelTasksResult",
    "SubmitModelResult",
    "SubmitModelResultCommand",
]
