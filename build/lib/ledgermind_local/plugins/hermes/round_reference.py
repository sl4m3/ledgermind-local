"""Source-round reference model used by plugin tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceReference:
    source_system: str
    source_instance_id: str
    source_profile_id: str
    source_session_id: str
    source_round_id: str
    first_message_id: str | None
    final_message_id: str | None
    message_ids: tuple[str, ...]
    source_digest: str | None
    source_schema_version: int
    resolver_version: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "source_instance_id": self.source_instance_id,
            "source_profile_id": self.source_profile_id,
            "source_session_id": self.source_session_id,
            "source_round_id": self.source_round_id,
            "first_message_id": self.first_message_id,
            "final_message_id": self.final_message_id,
            "message_ids": list(self.message_ids),
            "source_digest": self.source_digest,
            "source_schema_version": self.source_schema_version,
            "resolver_version": self.resolver_version,
        }
