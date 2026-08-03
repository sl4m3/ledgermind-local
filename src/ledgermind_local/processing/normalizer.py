"""Deterministic, model-free RawRound normalization and redaction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .models import NormalizedRound, NormalizedToolCall

_REDACT_INLINE = re.compile(
    r"(?i)(\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|password|secret|authorization)\b\s*[:=]\s*)([^\s,;]+)"
)
_REDACT_BEARER = re.compile(r"(?i)(\bbearer\s+)([^\s,;]+)")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "authorization",
    "credential",
    "private_key",
    "connection_string",
)


def redact_text(value: str) -> str:
    """Redact common credential-shaped values without model calls."""

    value = _REDACT_INLINE.sub(r"\1[REDACTED]", value)
    return _REDACT_BEARER.sub(r"\1[REDACTED]", value)


def redact_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None:
        normalized_key = key.casefold().replace("-", "_")
        if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
            return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(item_key): redact_value(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [redact_value(item) for item in value]
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return redact_text(content)
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray, str)):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(redact_text(text))
    return "".join(parts)


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _digest_payload(round_data: NormalizedRound) -> str:
    material = {
        "memory_space_id": round_data.memory_space_id,
        "source_system": round_data.source_system,
        "source_instance_id": round_data.source_instance_id,
        "source_profile_id": round_data.source_profile_id,
        "source_session_id": round_data.source_session_id,
        "source_round_id": round_data.source_round_id,
        "started_at": round_data.started_at,
        "completed_at": round_data.completed_at,
        "user_text": round_data.user_text,
        "assistant_text": round_data.assistant_text,
        "transcript": round_data.transcript,
        "tool_calls": [
            {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments_json": call.arguments_json,
            }
            for call in round_data.tool_calls
        ],
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_raw_round(payload: Mapping[str, Any], *, max_text_chars: int = 100_000) -> NormalizedRound:
    source = payload.get("source")
    body = payload.get("round")
    if not isinstance(source, Mapping) or not isinstance(body, Mapping):
        raise TypeError("RawRound must contain source and round objects")
    raw_events = body.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (bytes, bytearray, str)):
        raise TypeError("RawRound round.events must be an array")

    user_parts: list[str] = []
    assistant_parts: list[str] = []
    transcript_parts: list[str] = []
    tool_calls: list[NormalizedToolCall] = []
    events = sorted(
        (event for event in raw_events if isinstance(event, Mapping)),
        key=lambda event: (int(event.get("sequence", 0)), str(event.get("event_id", ""))),
    )
    for event in events:
        kind = str(event.get("kind", "message"))
        if kind == "tool_call":
            arguments = redact_value(event.get("arguments", {}))
            arguments_json = _bounded(
                json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                max_text_chars,
            )
            call = NormalizedToolCall(
                tool_call_id=str(event.get("tool_call_id", "")),
                tool_name=str(event.get("tool_name", "")),
                arguments_json=arguments_json,
            )
            tool_calls.append(call)
            transcript_parts.append(f"tool:{call.tool_name} args={call.arguments_json}")
            continue
        text = _content_text(event.get("content", []))
        if not text:
            continue
        role = str(event.get("role", ""))
        if role == "user":
            user_parts.append(text)
            transcript_parts.append(f"user: {text}")
        elif role == "assistant":
            assistant_parts.append(text)
            transcript_parts.append(f"assistant: {text}")

    normalized = NormalizedRound(
        memory_space_id=str(payload.get("memory_space_id", "")),
        source_system=str(source.get("system", "")),
        source_instance_id=str(source.get("instance_id", "")),
        source_profile_id=str(source.get("profile_id", "")),
        source_session_id=str(source.get("session_id", "")),
        source_round_id=str(source.get("round_id", "")),
        started_at=str(body.get("started_at", "")),
        completed_at=str(body.get("completed_at", "")),
        user_text=_bounded("\n".join(user_parts), max_text_chars),
        assistant_text=_bounded("\n".join(assistant_parts), max_text_chars),
        transcript=_bounded("\n".join(transcript_parts), max_text_chars),
        tool_calls=tuple(tool_calls),
        normalized_digest="",
    )
    return replace(normalized, normalized_digest=_digest_payload(normalized))
