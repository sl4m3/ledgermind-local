"""Stable Local-to-Core command contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


class CoreGatewayError(RuntimeError):
    """Base error for Core delivery failures."""


class TransientCoreError(CoreGatewayError):
    """Core is temporarily unavailable and the command may be retried."""


class CoreCapabilityError(CoreGatewayError):
    """The connected Core did not advertise a required IPC capability."""

    def __init__(
        self,
        *,
        requested: tuple[str, ...],
        missing_operations: tuple[str, ...] = (),
        missing_capabilities: tuple[str, ...] = (),
        expected_schema_version: int | None = None,
        advertised_schema_version: int | None = None,
    ) -> None:
        self.requested = requested
        self.missing_operations = missing_operations
        self.missing_capabilities = missing_capabilities
        self.expected_schema_version = expected_schema_version
        self.advertised_schema_version = advertised_schema_version
        details: list[str] = []
        if missing_operations:
            details.append("operations=" + ",".join(missing_operations))
        if missing_capabilities:
            details.append("capabilities=" + ",".join(missing_capabilities))
        if expected_schema_version is not None:
            details.append(
                "schema="
                + f"{advertised_schema_version or 'unavailable'}"
                + f" (required {expected_schema_version})"
            )
        suffix = "; ".join(details) if details else "no advertised support"
        super().__init__(f"Core capability validation failed: {suffix}")


class DomainRejectedError(CoreGatewayError):
    """Core rejected a validly delivered command for a domain reason."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _required(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _timestamp(value: object, name: str) -> str:
    text = _required(value, name)
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


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class IngestRawRoundCommand:
    """RawRound delivery envelope; the payload is loaded by the worker."""

    command_id: str
    idempotency_key: str
    memory_space_id: str
    raw_round_id: str
    raw_round: dict[str, Any]
    resolution_context: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "memory_space_id": self.memory_space_id,
            "raw_round_id": self.raw_round_id,
            "raw_round": self.raw_round,
        }
        if self.resolution_context is not None:
            payload["resolution_context"] = self.resolution_context
        return payload


@dataclass(frozen=True, slots=True)
class IngestRawRoundResult:
    accepted: bool
    duplicate: bool
    core_raw_round_id: str | None = None
    result_json: str | None = None


@dataclass(frozen=True, slots=True)
class CoreExecutionTask:
    """Strict technical execution envelope owned by the Local boundary.

    ``operation`` and ``operation_input`` are deliberately opaque.  Local only
    dispatches on ``task_kind`` and the Core remains the owner of domain
    semantics such as extraction and consolidation.
    """

    schema_version: int
    task_id: str
    task_kind: Literal["generate_json", "embed_texts"]
    operation: str
    profile_slot: Literal["operational", "background", "embedding"]
    memory_space_id: str
    expires_at: str
    lease: str | None
    model_request: dict[str, Any] | None
    embedding_request: dict[str, Any] | None
    operation_input: dict[str, Any] | None

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("schema_version must be 2")
        _required(self.task_id, "task_id")
        _required(self.operation, "operation")
        _required(self.memory_space_id, "memory_space_id")
        _timestamp(self.expires_at, "expires_at")
        if self.lease is not None:
            _timestamp(self.lease, "lease")
        for value, name in (
            (self.model_request, "model_request"),
            (self.embedding_request, "embedding_request"),
            (self.operation_input, "operation_input"),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be an object")
        if self.task_kind not in {"generate_json", "embed_texts"}:
            raise ValueError("task_kind must be generate_json or embed_texts")
        if self.task_kind == "generate_json":
            if self.profile_slot not in {"operational", "background"}:
                raise ValueError(
                    "generate_json task requires an operational or background profile slot"
                )
            if self.model_request is None or self.embedding_request is not None:
                raise ValueError("generate_json task has an invalid request shape")
            _strict_mapping(
                self.model_request,
                {"messages", "max_output_tokens", "response_format"},
                "model_request",
            )
            messages = self.model_request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("model_request.messages must be a non-empty array")
            max_output_tokens = self.model_request.get("max_output_tokens")
            if (
                isinstance(max_output_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or not 1 <= max_output_tokens <= 50_000
            ):
                raise TypeError("model_request.max_output_tokens must be an integer")
            response_format = self.model_request.get("response_format")
            if not isinstance(response_format, str) or response_format not in {
                "json_object",
                "text",
            }:
                raise ValueError("model_request.response_format is invalid")
            for message in messages:
                if not isinstance(message, dict):
                    raise TypeError("model_request.messages must contain objects")
                _strict_mapping(message, {"role", "content"}, "model message")
                _required(message.get("role"), "model message role")
                _required(message.get("content"), "model message content")
        else:
            if self.profile_slot != "embedding":
                raise ValueError("embed_texts task requires the embedding profile slot")
            if self.embedding_request is None or self.model_request is not None:
                raise ValueError("embed_texts task has an invalid request shape")
            _strict_mapping(
                self.embedding_request,
                {"texts", "purpose", "dimensions"},
                "embedding_request",
            )
            texts = self.embedding_request.get("texts")
            if not isinstance(texts, list) or not texts:
                raise ValueError("embedding_request.texts must be a non-empty array")
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("embedding_request.texts must contain text")
            _required(self.embedding_request.get("purpose"), "embedding purpose")
            dimensions = self.embedding_request.get("dimensions")
            if dimensions is not None and (
                isinstance(dimensions, bool)
                or not isinstance(dimensions, int)
                or not 1 <= dimensions <= 100_000
            ):
                raise TypeError("embedding_request.dimensions must be a positive integer")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CoreExecutionTask:
        _strict_mapping(
            payload,
            {
                "schema_version",
                "task_id",
                "task_kind",
                "operation",
                "profile_slot",
                "memory_space_id",
                "expires_at",
                "lease",
                "model_request",
                "embedding_request",
                "operation_input",
            },
            "object-facet execution task",
        )
        task_kind = payload.get("task_kind")
        if task_kind not in {"generate_json", "embed_texts"}:
            raise ValueError("task_kind must be generate_json or embed_texts")
        profile_slot = payload.get("profile_slot")
        if profile_slot not in {"operational", "background", "embedding"}:
            raise ValueError("profile_slot is invalid")
        model_request = payload.get("model_request")
        embedding_request = payload.get("embedding_request")
        operation_input = payload.get("operation_input")
        for value, name in (
            (model_request, "model_request"),
            (embedding_request, "embedding_request"),
            (operation_input, "operation_input"),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be an object")
        return cls(
            schema_version=_non_negative_int(payload.get("schema_version"), "schema_version"),
            task_id=_required(payload.get("task_id"), "task_id"),
            task_kind=task_kind,
            operation=_required(payload.get("operation"), "operation"),
            profile_slot=profile_slot,
            memory_space_id=_required(
                payload.get("memory_space_id"), "memory_space_id"
            ),
            expires_at=_required(payload.get("expires_at"), "expires_at"),
            lease=(
                _required(payload["lease"], "lease")
                if payload.get("lease") is not None
                else None
            ),
            model_request=dict(model_request) if model_request is not None else None,
            embedding_request=(
                dict(embedding_request) if embedding_request is not None else None
            ),
            operation_input=(
                dict(operation_input) if operation_input is not None else None
            ),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "operation": self.operation,
            "profile_slot": self.profile_slot,
            "memory_space_id": self.memory_space_id,
            "expires_at": self.expires_at,
            "model_request": self.model_request,
            "embedding_request": self.embedding_request,
            "operation_input": self.operation_input,
        }
        if self.lease is not None:
            payload["lease"] = self.lease
        return payload


@dataclass(frozen=True, slots=True)
class CoreExecutionResult:
    """Strict generic result envelope sent back to Core."""

    task_id: str
    task_kind: Literal["generate_json", "embed_texts"]
    status: Literal["completed"]
    operation: str
    operation_input: dict[str, Any] | None
    output: dict[str, Any] | None
    embedding_result: dict[str, Any] | None
    egress_audit: dict[str, Any]
    error_code: str | None = None

    def __post_init__(self) -> None:
        _required(self.task_id, "result.task_id")
        _required(self.operation, "result.operation")
        if self.task_kind not in {"generate_json", "embed_texts"}:
            raise ValueError("result.task_kind is invalid")
        if self.status != "completed":
            raise ValueError("result.status must be completed")
        if self.operation_input is not None and not isinstance(self.operation_input, dict):
            raise TypeError("result.operation_input must be an object")
        if self.output is not None and not isinstance(self.output, dict):
            raise TypeError("result.output must be an object")
        if self.embedding_result is not None and not isinstance(self.embedding_result, dict):
            raise TypeError("result.embedding_result must be an object")
        if not isinstance(self.egress_audit, dict):
            raise TypeError("result.egress_audit must be an object")
        if self.task_kind == "generate_json":
            if self.output is None or self.embedding_result is not None:
                raise ValueError("generate_json result has an invalid output shape")
        elif self.embedding_result is None or self.output is not None:
            raise ValueError("embed_texts result has an invalid output shape")
        if self.error_code is not None:
            _required(self.error_code, "result.error_code")

    def to_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "status": self.status,
            "operation": self.operation,
            "operation_input": self.operation_input,
            "output": self.output,
            "embedding_result": self.embedding_result,
            "egress_audit": self.egress_audit,
            "error_code": self.error_code,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CoreExecutionResult:
        _strict_mapping(
            payload,
            {
                "task_id",
                "task_kind",
                "status",
                "operation",
                "operation_input",
                "output",
                "embedding_result",
                "egress_audit",
                "error_code",
            },
            "object-facet execution result",
        )
        task_kind = payload.get("task_kind")
        if task_kind not in {"generate_json", "embed_texts"}:
            raise ValueError("result.task_kind is invalid")
        if payload.get("status") != "completed":
            raise ValueError("result.status must be completed")
        operation_input = payload.get("operation_input")
        output = payload.get("output")
        embedding_result = payload.get("embedding_result")
        audit = payload.get("egress_audit")
        for value, name in (
            (operation_input, "result.operation_input"),
            (output, "result.output"),
            (embedding_result, "result.embedding_result"),
            (audit, "result.egress_audit"),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be an object")
        if not isinstance(audit, dict):
            raise TypeError("result.egress_audit must be an object")
        if task_kind == "generate_json":
            if not isinstance(output, dict) or embedding_result is not None:
                raise ValueError("generate_json result has an invalid output shape")
        elif not isinstance(embedding_result, dict) or output is not None:
            raise ValueError("embed_texts result has an invalid output shape")
        return cls(
            task_id=_required(payload.get("task_id"), "result.task_id"),
            task_kind=task_kind,
            status="completed",
            operation=_required(payload.get("operation"), "result.operation"),
            operation_input=(
                dict(operation_input) if operation_input is not None else None
            ),
            output=dict(output) if output is not None else None,
            embedding_result=(
                dict(embedding_result) if embedding_result is not None else None
            ),
            egress_audit=dict(audit),
            error_code=(
                _required(payload["error_code"], "result.error_code")
                if payload.get("error_code") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PollExecutionTasksCommand:
    request_id: str
    memory_space_id: str
    worker_id: str
    limit: int = 10
    lease_seconds: int = 300

    def to_payload(self) -> dict[str, object]:
        return {
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "limit": self.limit,
            "lease_seconds": self.lease_seconds,
        }


@dataclass(frozen=True, slots=True)
class PollExecutionTasksResult:
    tasks: tuple[dict[str, Any], ...]
    has_more: bool = False


@dataclass(frozen=True, slots=True)
class SubmitExecutionResultCommand:
    request_id: str
    task_id: str
    memory_space_id: str
    worker_id: str
    result: dict[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class SubmitExecutionResult:
    accepted: bool
    duplicate: bool = False
    status: str = "accepted"


@dataclass(frozen=True, slots=True)
class FailExecutionTaskCommand:
    request_id: str
    task_id: str
    memory_space_id: str
    worker_id: str
    error_code: str
    retryable: bool
    retry_after_seconds: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "memory_space_id": self.memory_space_id,
            "worker_id": self.worker_id,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass(frozen=True, slots=True)
class FailExecutionTaskResult:
    released: bool
    retry_scheduled: bool = False
    terminal: bool = False
    status: str = "failed"


@dataclass(frozen=True, slots=True)
class RetrieveContextCommand:
    request_id: str
    memory_space_id: str
    query_text: str
    query_embedding: tuple[float, ...]
    embedding_model_id: str = "retrieval-embedder"
    embedding_model_version: str = "1"
    limit: int = 5
    project_id: str | None = None
    repository_id: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    related_object_ids: tuple[str, ...] = ()
    requested_facets: tuple[str, ...] = ()
    explanation_level: str = "compact"

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _required(self.memory_space_id, "memory_space_id")
        _required(self.query_text, "query_text")
        if not self.query_embedding:
            raise ValueError("query_embedding must not be empty")
        if any(
            not isinstance(component, (int, float))
            or isinstance(component, bool)
            or not math.isfinite(float(component))
            for component in self.query_embedding
        ):
            raise ValueError("query_embedding must contain finite values")
        _required(self.embedding_model_id, "embedding_model_id")
        _required(self.embedding_model_version, "embedding_model_version")
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if self.repository_id is not None and self.project_id is None:
            raise ValueError("repository_id requires project_id")
        if self.explanation_level not in {"compact", "none"}:
            raise ValueError("explanation_level must be compact or none")
        if len(set(self.related_object_ids)) != len(self.related_object_ids):
            raise ValueError("related_object_ids must be unique")
        if len(set(self.requested_facets)) != len(self.requested_facets):
            raise ValueError("requested_facets must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "memory_space_id": self.memory_space_id,
            "query_text": self.query_text,
            "query_embedding": list(self.query_embedding),
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_version": self.embedding_model_version,
            "limit": self.limit,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "related_object_ids": list(self.related_object_ids),
            "requested_facets": list(self.requested_facets),
            "explanation_level": self.explanation_level,
        }


@dataclass(frozen=True, slots=True)
class RetrieveContextResult:
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecordRetrievalOutcomeCommand:
    request_id: str
    retrieval_request_id: str
    candidate_value_ids: tuple[str, ...]
    delivered_value_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _required(self.retrieval_request_id, "retrieval_request_id")
        if not self.candidate_value_ids:
            raise ValueError("candidate_value_ids must not be empty")
        if len(self.candidate_value_ids) > 100:
            raise ValueError("candidate_value_ids must not exceed 100 entries")
        if len(self.delivered_value_ids) > 100:
            raise ValueError("delivered_value_ids must not exceed 100 entries")
        if len(set(self.candidate_value_ids)) != len(self.candidate_value_ids):
            raise ValueError("candidate_value_ids must be unique")
        if len(set(self.delivered_value_ids)) != len(self.delivered_value_ids):
            raise ValueError("delivered_value_ids must be unique")
        if not set(self.delivered_value_ids).issubset(set(self.candidate_value_ids)):
            raise ValueError("delivered_value_ids must be a subset of candidates")
        for value_id in (*self.candidate_value_ids, *self.delivered_value_ids):
            _required(value_id, "value_id")

    def to_payload(self) -> dict[str, object]:
        return {
            "retrieval_request_id": self.retrieval_request_id,
            "candidate_value_ids": list(self.candidate_value_ids),
            "delivered_value_ids": list(self.delivered_value_ids),
        }


@dataclass(frozen=True, slots=True)
class CoreHealth:
    healthy: bool
    backend: str
    detail: str | None = None
    protocol_version: int | None = None
    schema_version: int | None = None


@dataclass(frozen=True, slots=True)
class RunControlMaintenanceCommand:
    request_id: str

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")

    def to_payload(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class ControlMaintenanceResult:
    status: str
    memory_echoes_reconciled: int
    stats_rebuilt: int
    stale_jobs_recovered: int
    findings_created: int
    duplicate_object_findings: int
    missing_card_embeddings: int
    missing_facet_embeddings: int
    integrity_errors: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ControlMaintenanceResult:
        _strict_mapping(
            payload,
            {
                "status",
                "memory_echoes_reconciled",
                "stats_rebuilt",
                "stale_jobs_recovered",
                "findings_created",
                "duplicate_object_findings",
                "missing_card_embeddings",
                "missing_facet_embeddings",
                "integrity_errors",
            },
            "control maintenance result",
        )
        if payload.get("status") != "completed":
            raise ValueError("control maintenance status must be completed")
        return cls(
            status="completed",
            memory_echoes_reconciled=_non_negative_int(
                payload.get("memory_echoes_reconciled"), "memory_echoes_reconciled"
            ),
            stats_rebuilt=_non_negative_int(payload.get("stats_rebuilt"), "stats_rebuilt"),
            stale_jobs_recovered=_non_negative_int(
                payload.get("stale_jobs_recovered"), "stale_jobs_recovered"
            ),
            findings_created=_non_negative_int(
                payload.get("findings_created"), "findings_created"
            ),
            duplicate_object_findings=_non_negative_int(
                payload.get("duplicate_object_findings"), "duplicate_object_findings"
            ),
            missing_card_embeddings=_non_negative_int(
                payload.get("missing_card_embeddings"), "missing_card_embeddings"
            ),
            missing_facet_embeddings=_non_negative_int(
                payload.get("missing_facet_embeddings"), "missing_facet_embeddings"
            ),
            integrity_errors=_non_negative_int(payload.get("integrity_errors"), "integrity_errors"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "memory_echoes_reconciled": self.memory_echoes_reconciled,
            "stats_rebuilt": self.stats_rebuilt,
            "stale_jobs_recovered": self.stale_jobs_recovered,
            "findings_created": self.findings_created,
            "duplicate_object_findings": self.duplicate_object_findings,
            "missing_card_embeddings": self.missing_card_embeddings,
            "missing_facet_embeddings": self.missing_facet_embeddings,
            "integrity_errors": self.integrity_errors,
        }


@dataclass(frozen=True, slots=True)
class ObjectFacetStatistics:
    object_count: int
    active_value_count: int
    superseded_value_count: int
    operational_backlog: int
    background_backlog: int
    embedding_backlog: int
    integrity_finding_count: int
    missing_card_embeddings: int | None = None
    missing_facet_embeddings: int | None = None
    legacy_digest_upgrade_required: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ObjectFacetStatistics:
        _strict_mapping(
            payload,
            {
                "object_count",
                "active_value_count",
                "superseded_value_count",
                "operational_backlog",
                "background_backlog",
                "embedding_backlog",
                "integrity_finding_count",
                "missing_card_embeddings",
                "missing_facet_embeddings",
                "legacy_digest_upgrade_required",
            },
            "object-facet statistics",
        )

        def optional_count(name: str) -> int | None:
            value = payload.get(name)
            return None if value is None else _non_negative_int(value, name)

        legacy = payload.get("legacy_digest_upgrade_required", False)
        if not isinstance(legacy, bool):
            raise TypeError("legacy_digest_upgrade_required must be a boolean")
        return cls(
            object_count=_non_negative_int(payload.get("object_count"), "object_count"),
            active_value_count=_non_negative_int(
                payload.get("active_value_count"), "active_value_count"
            ),
            superseded_value_count=_non_negative_int(
                payload.get("superseded_value_count"), "superseded_value_count"
            ),
            operational_backlog=_non_negative_int(
                payload.get("operational_backlog"), "operational_backlog"
            ),
            background_backlog=_non_negative_int(
                payload.get("background_backlog"), "background_backlog"
            ),
            embedding_backlog=_non_negative_int(
                payload.get("embedding_backlog"), "embedding_backlog"
            ),
            integrity_finding_count=_non_negative_int(
                payload.get("integrity_finding_count"), "integrity_finding_count"
            ),
            missing_card_embeddings=optional_count("missing_card_embeddings"),
            missing_facet_embeddings=optional_count("missing_facet_embeddings"),
            legacy_digest_upgrade_required=legacy,
        )


__all__ = [
    "ControlMaintenanceResult",
    "CoreCapabilityError",
    "CoreExecutionResult",
    "CoreExecutionTask",
    "CoreGatewayError",
    "CoreHealth",
    "DomainRejectedError",
    "FailExecutionTaskCommand",
    "FailExecutionTaskResult",
    "IngestRawRoundCommand",
    "IngestRawRoundResult",
    "ObjectFacetStatistics",
    "PollExecutionTasksCommand",
    "PollExecutionTasksResult",
    "RecordRetrievalOutcomeCommand",
    "RetrieveContextCommand",
    "RetrieveContextResult",
    "RunControlMaintenanceCommand",
    "SubmitExecutionResult",
    "SubmitExecutionResultCommand",
    "TransientCoreError",
]
