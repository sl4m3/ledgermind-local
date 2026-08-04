"""Safe Core projection event contracts crossing the Local process boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

CORE_PROJECTION_UPSERT = "knowledge_projection_upsert"
CORE_PROJECTION_DELETE = "knowledge_projection_delete"


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _projection_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("projection_version must be a positive integer")
    return value


def _strict_fields(payload: dict[str, Any], expected: set[str]) -> None:
    unknown = sorted(set(payload) - expected)
    if unknown:
        raise ValueError(f"unknown projection payload fields: {', '.join(unknown)}")
    missing = sorted(expected - set(payload))
    if missing:
        raise ValueError(f"missing projection payload fields: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class ProjectionUpsertPayload:
    knowledge_id: str
    memory_space_id: str
    title: str
    target: str
    statement: str
    projection_version: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ProjectionUpsertPayload:
        expected = {
            "knowledge_id",
            "memory_space_id",
            "title",
            "target",
            "statement",
            "projection_version",
        }
        _strict_fields(payload, expected)
        return cls(
            knowledge_id=_required_text(payload["knowledge_id"], "knowledge_id"),
            memory_space_id=_required_text(
                payload["memory_space_id"], "memory_space_id"
            ),
            title=_required_text(payload["title"], "title"),
            target=_required_text(payload["target"], "target"),
            statement=_required_text(payload["statement"], "statement"),
            projection_version=_projection_version(payload["projection_version"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "knowledge_id": self.knowledge_id,
            "memory_space_id": self.memory_space_id,
            "title": self.title,
            "target": self.target,
            "statement": self.statement,
            "projection_version": self.projection_version,
        }


@dataclass(frozen=True, slots=True)
class ProjectionDeletePayload:
    knowledge_id: str
    memory_space_id: str
    projection_version: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ProjectionDeletePayload:
        expected = {"knowledge_id", "memory_space_id", "projection_version"}
        _strict_fields(payload, expected)
        return cls(
            knowledge_id=_required_text(payload["knowledge_id"], "knowledge_id"),
            memory_space_id=_required_text(
                payload["memory_space_id"], "memory_space_id"
            ),
            projection_version=_projection_version(payload["projection_version"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "knowledge_id": self.knowledge_id,
            "memory_space_id": self.memory_space_id,
            "projection_version": self.projection_version,
        }


@dataclass(frozen=True, slots=True)
class CoreProjectionEvent:
    event_id: str
    memory_space_id: str
    aggregate_id: str
    event_type: str
    payload_json: str
    occurred_at: str

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        _required_text(self.memory_space_id, "memory_space_id")
        _required_text(self.aggregate_id, "aggregate_id")
        _required_text(self.event_type, "event_type")
        _required_text(self.payload_json, "payload_json")
        _required_text(self.occurred_at, "occurred_at")

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> CoreProjectionEvent:
        expected = {
            "event_id",
            "memory_space_id",
            "aggregate_id",
            "event_type",
            "payload",
            "occurred_at",
        }
        _strict_fields(payload, expected)
        raw_payload = payload["payload"]
        if not isinstance(raw_payload, dict):
            raise TypeError("projection event payload must be an object")
        return cls(
            event_id=_required_text(payload["event_id"], "event_id"),
            memory_space_id=_required_text(
                payload["memory_space_id"], "memory_space_id"
            ),
            aggregate_id=_required_text(payload["aggregate_id"], "aggregate_id"),
            event_type=_required_text(payload["event_type"], "event_type"),
            payload_json=json.dumps(raw_payload, ensure_ascii=False, separators=(",", ":")),
            occurred_at=_required_text(payload["occurred_at"], "occurred_at"),
        )

    def wire_payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "memory_space_id": self.memory_space_id,
            "aggregate_id": self.aggregate_id,
            "event_type": self.event_type,
            "payload": json.loads(self.payload_json),
            "occurred_at": self.occurred_at,
        }

    def parse_payload(self) -> ProjectionUpsertPayload | ProjectionDeletePayload:
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("projection event payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("projection event payload must be an object")
        if self.event_type == CORE_PROJECTION_UPSERT:
            parsed: ProjectionUpsertPayload | ProjectionDeletePayload = (
                ProjectionUpsertPayload.from_payload(payload)
            )
        elif self.event_type == CORE_PROJECTION_DELETE:
            try:
                parsed = ProjectionDeletePayload.from_payload(payload)
            except ValueError as exc:
                if {"title", "target", "statement"}.intersection(payload):
                    raise ValueError("projection payload does not match event type") from exc
                raise
        else:
            raise ValueError(f"unsupported Core projection event type: {self.event_type}")
        if parsed.memory_space_id != self.memory_space_id:
            raise ValueError("projection payload memory_space_id does not match event")
        if parsed.knowledge_id != self.aggregate_id:
            raise ValueError("projection payload knowledge_id does not match event")
        return parsed


@dataclass(frozen=True, slots=True)
class PollProjectionEventsResult:
    events: tuple[CoreProjectionEvent, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class PollProjectionEventsCommand:
    request_id: str
    memory_space_id: str
    consumer_id: str
    after_event_id: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.memory_space_id, "memory_space_id")
        _required_text(self.consumer_id, "consumer_id")
        if not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

    def to_payload(self) -> dict[str, object]:
        return {
            "memory_space_id": self.memory_space_id,
            "consumer_id": self.consumer_id,
            "after_event_id": self.after_event_id,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class AckProjectionEventsCommand:
    request_id: str
    consumer_id: str
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.consumer_id, "consumer_id")
        if not self.event_ids or any(not isinstance(value, str) or not value for value in self.event_ids):
            raise ValueError("event_ids must contain at least one non-empty ID")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("event_ids must be unique")

    def to_payload(self) -> dict[str, object]:
        return {"consumer_id": self.consumer_id, "event_ids": list(self.event_ids)}


@dataclass(frozen=True, slots=True)
class AckProjectionEventsResult:
    acknowledged: tuple[str, ...]


__all__ = [
    "CORE_PROJECTION_DELETE",
    "CORE_PROJECTION_UPSERT",
    "AckProjectionEventsCommand",
    "AckProjectionEventsResult",
    "CoreProjectionEvent",
    "PollProjectionEventsCommand",
    "PollProjectionEventsResult",
    "ProjectionDeletePayload",
    "ProjectionUpsertPayload",
]
