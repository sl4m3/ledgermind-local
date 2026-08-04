"""Deterministic, model-free RawRound normalization and redaction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .models import NormalizedRound, NormalizedToolInteraction

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
        return {
            str(item_key): redact_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [redact_value(item) for item in value]
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return redact_text(content)
    if not isinstance(content, Sequence) or isinstance(
        content, (bytes, bytearray, str)
    ):
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
        "tool_interactions": [
            {
                "tool_call_id": interaction.tool_call_id,
                "tool_name": interaction.tool_name,
                "arguments_json": interaction.arguments_json,
                "result_text": interaction.result_text,
                "result_json": interaction.result_json,
                "status": interaction.status,
                "error_text": interaction.error_text,
                "source_call_event_id": interaction.source_call_event_id,
                "source_result_event_id": interaction.source_result_event_id,
            }
            for interaction in round_data.tool_interactions
        ],
        "source_event_ids": list(round_data.source_event_ids),
        "normalizer_version": round_data.normalizer_version,
    }
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _content_parts(content: Any) -> tuple[str, str | None]:
    if not isinstance(content, Sequence) or isinstance(
        content, (bytes, bytearray, str)
    ):
        return "", None
    text_parts: list[str] = []
    json_parts: list[Any] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            text_parts.append(redact_text(str(part["text"])))
        elif part_type == "json":
            json_parts.append(redact_value(part.get("data")))
        elif part_type == "reference" and isinstance(part.get("uri"), str):
            text_parts.append(redact_text(str(part["uri"])))
    result_json: str | None = None
    if json_parts:
        encoded = json_parts[0] if len(json_parts) == 1 else json_parts
        result_json = json.dumps(
            encoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return "".join(text_parts), result_json


def _result_transcript(
    *,
    tool_call_id: str,
    status: str,
    result_text: str,
    result_json: str | None,
) -> str:
    value = result_text or (f"json={result_json}" if result_json is not None else "")
    label = "error" if status in {"error", "cancelled"} else "result"
    suffix = f" {label}={value}" if value else ""
    return f"tool_result:{tool_call_id} status={status}{suffix}"


def normalize_raw_round(
    payload: Mapping[str, Any],
    *,
    max_text_chars: int = 100_000,
    normalizer_version: int = 1,
) -> NormalizedRound:
    if normalizer_version < 1:
        raise ValueError("normalizer_version must be positive")
    source = payload.get("source")
    body = payload.get("round")
    if not isinstance(source, Mapping) or not isinstance(body, Mapping):
        raise TypeError("RawRound must contain source and round objects")
    raw_events = body.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(
        raw_events, (bytes, bytearray, str)
    ):
        raise TypeError("RawRound round.events must be an array")

    user_parts: list[str] = []
    assistant_parts: list[str] = []
    transcript_parts: list[str] = []
    interactions: dict[str, NormalizedToolInteraction] = {}
    result_event_ids: set[str] = set()
    events = sorted(
        (event for event in raw_events if isinstance(event, Mapping)),
        key=lambda event: (
            int(event.get("sequence", 0)),
            str(event.get("event_id", "")),
        ),
    )
    source_event_ids = tuple(str(event.get("event_id", "")) for event in events)
    for event in events:
        kind = str(event.get("kind", "message"))
        if kind == "tool_call":
            tool_call_id = str(event.get("tool_call_id", ""))
            tool_name = str(event.get("tool_name", ""))
            if not tool_call_id or not tool_name:
                raise ValueError("tool_call requires tool_call_id and tool_name")
            if tool_call_id in interactions:
                raise ValueError(f"duplicate tool_call: {tool_call_id}")
            arguments = redact_value(event.get("arguments", {}))
            arguments_json = _bounded(
                json.dumps(
                    arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                max_text_chars,
            )
            interaction = NormalizedToolInteraction(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments_json=arguments_json,
                result_text="",
                result_json=None,
                status="unknown",
                error_text="",
                source_call_event_id=str(event.get("event_id", "")),
                source_result_event_id=None,
            )
            interactions[tool_call_id] = interaction
            transcript_parts.append(
                f"tool:{interaction.tool_name} args={interaction.arguments_json}"
            )
            continue
        if kind == "tool_result":
            tool_call_id = str(event.get("tool_call_id", ""))
            if tool_call_id not in interactions:
                raise ValueError(
                    f"tool_result without a matching tool_call: {tool_call_id}"
                )
            event_id = str(event.get("event_id", ""))
            if tool_call_id in result_event_ids:
                raise ValueError(f"duplicate tool_result: {tool_call_id}")
            result_event_ids.add(tool_call_id)
            result_text, result_json = _content_parts(event.get("content", []))
            result_text = _bounded(result_text, max_text_chars)
            result_json = (
                _bounded(result_json, max_text_chars)
                if result_json is not None
                else None
            )
            status = str(event.get("status", "unknown"))
            error_value = event.get("error")
            error_text = (
                redact_text(str(error_value))
                if error_value is not None
                else result_text
                if status in {"error", "cancelled"}
                else ""
            )
            interactions[tool_call_id] = replace(
                interactions[tool_call_id],
                result_text=result_text,
                result_json=result_json,
                status=status,
                error_text=_bounded(error_text, max_text_chars),
                source_result_event_id=event_id,
            )
            transcript_parts.append(
                _result_transcript(
                    tool_call_id=tool_call_id,
                    status=status,
                    result_text=result_text,
                    result_json=result_json,
                )
            )
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
        tool_interactions=tuple(interactions.values()),
        normalized_digest="",
        source_event_ids=source_event_ids,
        normalizer_version=normalizer_version,
    )
    return replace(normalized, normalized_digest=_digest_payload(normalized))
