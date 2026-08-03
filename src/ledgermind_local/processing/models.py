"""Processing data structures shared by normalizer and generators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    tool_call_id: str
    tool_name: str
    arguments_json: str


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
    tool_calls: tuple[NormalizedToolCall, ...]
    normalized_digest: str
