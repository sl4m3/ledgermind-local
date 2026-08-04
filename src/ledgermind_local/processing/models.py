"""Processing data structures shared by normalizer and generators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedToolInteraction:
    tool_call_id: str
    tool_name: str
    arguments_json: str
    result_text: str
    result_json: str | None
    status: str
    error_text: str
    source_call_event_id: str
    source_result_event_id: str | None


@dataclass(frozen=True, slots=True)
class NormalizedRound:
    memory_space_id: str
    source_system: str
    source_instance_id: str
    source_profile_id: str
    source_session_id: str
    source_round_id: str
    started_at: str
    completed_at: str
    user_text: str
    assistant_text: str
    transcript: str
    tool_interactions: tuple[NormalizedToolInteraction, ...]
    normalized_digest: str
    source_event_ids: tuple[str, ...] = ()
    normalizer_version: int = 1
