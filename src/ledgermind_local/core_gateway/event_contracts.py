"""Stable Local projection event names used by projection adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Structural event contract consumed by Local durable outbox adapters."""

    event_id: str
    event_type: str
    aggregate_id: str
    memory_space_id: str
    payload_json: str
    occurred_at: datetime


class KnowledgeCreated:
    EVENT_NAME = "knowledge.created"


class KnowledgeDeleted:
    EVENT_NAME = "knowledge.deleted"


class KnowledgeSuperseded:
    EVENT_NAME = "knowledge.superseded"


__all__ = [
    "DomainEvent",
    "KnowledgeCreated",
    "KnowledgeDeleted",
    "KnowledgeSuperseded",
]
