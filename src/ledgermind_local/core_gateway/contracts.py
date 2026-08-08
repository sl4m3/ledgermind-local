"""Versioned Local-to-Core command contracts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    ) -> None:
        self.requested = requested
        self.missing_operations = missing_operations
        self.missing_capabilities = missing_capabilities
        details: list[str] = []
        if missing_operations:
            details.append("operations=" + ",".join(missing_operations))
        if missing_capabilities:
            details.append("capabilities=" + ",".join(missing_capabilities))
        suffix = "; ".join(details) if details else "no advertised support"
        super().__init__(f"Core capability validation failed: {suffix}")


class DomainRejectedError(CoreGatewayError):
    """Core rejected a validly delivered command for a domain reason."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _digest(value: str, name: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must match sha256:<64 hex>")
    return value


@dataclass(frozen=True, slots=True)
class HypothesisEvidence:
    source_system: str
    source_instance_id: str
    source_profile_id: str
    source_session_id: str
    source_round_id: str
    raw_round_digest: str
    normalized_round_digest: str
    source_event_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "source_system": self.source_system,
            "source_instance_id": self.source_instance_id,
            "source_profile_id": self.source_profile_id,
            "source_session_id": self.source_session_id,
            "source_round_id": self.source_round_id,
            "raw_round_digest": self.raw_round_digest,
            "normalized_round_digest": self.normalized_round_digest,
            "source_event_ids": list(self.source_event_ids),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HypothesisEvidence:
        return cls(
            source_system=_required(str(payload["source_system"]), "source_system"),
            source_instance_id=_required(
                str(payload["source_instance_id"]), "source_instance_id"
            ),
            source_profile_id=_required(
                str(payload["source_profile_id"]), "source_profile_id"
            ),
            source_session_id=_required(
                str(payload["source_session_id"]), "source_session_id"
            ),
            source_round_id=_required(
                str(payload["source_round_id"]), "source_round_id"
            ),
            raw_round_digest=_digest(
                str(payload["raw_round_digest"]), "raw_round_digest"
            ),
            normalized_round_digest=_digest(
                str(payload["normalized_round_digest"]), "normalized_round_digest"
            ),
            source_event_ids=tuple(
                str(item) for item in payload.get("source_event_ids", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class HypothesisExtraction:
    provider: str
    model: str
    prompt_version: int
    schema_version: int
    completed_at: str

    def to_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HypothesisExtraction:
        return cls(
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            prompt_version=int(payload["prompt_version"]),
            schema_version=int(payload["schema_version"]),
            completed_at=str(payload["completed_at"]),
        )


@dataclass(frozen=True, slots=True)
class HypothesisPayload:
    hypothesis_id: str
    content_digest: str
    title: str
    target: str
    statement: str
    rationale: str
    result: str
    artifacts: tuple[str, ...]
    evidence: HypothesisEvidence
    extraction: HypothesisExtraction

    def to_payload(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "content_digest": self.content_digest,
            "title": self.title,
            "target": self.target,
            "statement": self.statement,
            "rationale": self.rationale,
            "result": self.result,
            "artifacts": list(self.artifacts),
            "evidence": self.evidence.to_payload(),
            "extraction": self.extraction.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HypothesisPayload:
        return cls(
            hypothesis_id=_required(str(payload["hypothesis_id"]), "hypothesis_id"),
            content_digest=_digest(str(payload["content_digest"]), "content_digest"),
            title=_required(str(payload["title"]), "title"),
            target=_required(str(payload["target"]), "target"),
            statement=_required(str(payload["statement"]), "statement"),
            rationale=str(payload.get("rationale", "")),
            result=str(payload.get("result", "")),
            artifacts=tuple(str(item) for item in payload.get("artifacts", [])),
            evidence=HypothesisEvidence.from_payload(dict(payload["evidence"])),
            extraction=HypothesisExtraction.from_payload(dict(payload["extraction"])),
        )


@dataclass(frozen=True, slots=True)
class AcceptHypothesisCommand:
    protocol_version: int
    command_id: str
    idempotency_key: str
    memory_space_id: str
    hypothesis: HypothesisPayload

    def __post_init__(self) -> None:
        if self.protocol_version != 1:
            raise ValueError("unsupported Core command protocol version")
        _required(self.command_id, "command_id")
        _digest(self.idempotency_key, "idempotency_key")
        _required(self.memory_space_id, "memory_space_id")

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "memory_space_id": self.memory_space_id,
            "hypothesis": self.hypothesis.to_payload(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AcceptHypothesisCommand:
        return cls(
            protocol_version=int(payload["protocol_version"]),
            command_id=str(payload["command_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            memory_space_id=str(payload["memory_space_id"]),
            hypothesis=HypothesisPayload.from_payload(dict(payload["hypothesis"])),
        )

    @classmethod
    def from_json(cls, payload_json: str) -> AcceptHypothesisCommand:
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise TypeError("Core command payload must be a JSON object")
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class AcceptHypothesisResult:
    accepted: bool
    duplicate: bool
    core_reference_id: str | None = None
    result_json: str | None = None


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
    operational_job_id: str | None = None
    result_json: str | None = None


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
class RetrieveContextV2Command:
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
        if any(not math.isfinite(float(component)) for component in self.query_embedding):
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
class RetrieveContextV2Result:
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecordRetrievalOutcomeV2Command:
    request_id: str
    retrieval_request_id: str
    candidate_value_ids: tuple[str, ...]
    delivered_value_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "retrieval_request_id": self.retrieval_request_id,
            "candidate_value_ids": list(self.candidate_value_ids),
            "delivered_value_ids": list(self.delivered_value_ids),
        }


@dataclass(frozen=True, slots=True)
class RetrieveContextCommand:
    request_id: str
    memory_space_id: str
    query: str
    limit: int = 5
    candidate_ids: tuple[str, ...] = ()
    candidate_scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _required(self.memory_space_id, "memory_space_id")
        _required(self.query, "query")
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        score_ids = tuple(candidate_id for candidate_id, _ in self.candidate_scores)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("candidate_scores IDs must be unique")
        if set(score_ids) - set(self.candidate_ids):
            raise ValueError("candidate_scores must refer to candidate_ids")


@dataclass(frozen=True, slots=True)
class ContextViewItem:
    knowledge_id: str
    title: str
    target: str
    statement: str
    relevance: float


@dataclass(frozen=True, slots=True)
class ContextViewResult:
    items: tuple[ContextViewItem, ...]
    api_version: str = "1"

    def to_payload(self) -> dict[str, object]:
        return {
            "api_version": self.api_version,
            "items": [
                {
                    "knowledge_id": item.knowledge_id,
                    "title": item.title,
                    "target": item.target,
                    "statement": item.statement,
                    "relevance": item.relevance,
                }
                for item in self.items
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_core_json(cls, payload_json: str) -> ContextViewResult:
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise TypeError("context result must be an object")
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise TypeError("context items must be a list")
        items: list[ContextViewItem] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise TypeError("context item must be an object")
            relevance = raw_item.get("relevance", raw_item.get("score", 0.0))
            if (
                not isinstance(relevance, (int, float))
                or not 0.0 <= float(relevance) <= 1.0
            ):
                raise ValueError("context relevance must be between 0 and 1")
            values: dict[str, str] = {}
            for name in ("knowledge_id", "title", "target", "statement"):
                value = raw_item.get(name)
                if not isinstance(value, str) or not value:
                    raise ValueError("context item contains invalid public fields")
                values[name] = value
            items.append(
                ContextViewItem(
                    knowledge_id=values["knowledge_id"],
                    title=values["title"],
                    target=values["target"],
                    statement=values["statement"],
                    relevance=float(relevance),
                )
            )
        return cls(items=tuple(items), api_version=str(payload.get("api_version", "1")))


@dataclass(frozen=True, slots=True)
class RecordContextUsageCommand:
    request_id: str
    memory_space_id: str
    item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoreHealth:
    healthy: bool
    backend: str
    detail: str | None = None


__all__ = [
    "AcceptHypothesisCommand",
    "AcceptHypothesisResult",
    "ContextViewItem",
    "ContextViewResult",
    "CoreCapabilityError",
    "CoreGatewayError",
    "CoreHealth",
    "DomainRejectedError",
    "FailExecutionTaskCommand",
    "FailExecutionTaskResult",
    "HypothesisEvidence",
    "HypothesisExtraction",
    "HypothesisPayload",
    "IngestRawRoundCommand",
    "IngestRawRoundResult",
    "PollExecutionTasksCommand",
    "PollExecutionTasksResult",
    "RecordContextUsageCommand",
    "RecordRetrievalOutcomeV2Command",
    "RetrieveContextCommand",
    "RetrieveContextV2Command",
    "RetrieveContextV2Result",
    "SubmitExecutionResult",
    "SubmitExecutionResultCommand",
    "TransientCoreError",
]
