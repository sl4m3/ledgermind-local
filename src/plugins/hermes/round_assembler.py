"""Helpers for extracting a completed Hermes round."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_ROUND_LIMIT_TEXT = 4_000


def _coerce_messages(context: object) -> list[dict[str, Any]]:
    if not isinstance(context, list):
        return []
    return [entry for entry in context if isinstance(entry, dict)]


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_tool_args(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, int, float, bool, type(None))):
        return json.dumps(value, ensure_ascii=False)
    return _to_text(value)


def _to_message_text(message: dict[str, Any]) -> str:
    return _to_text(message.get("api_content", message.get("content"))).strip()


def _normalize_tool_calls(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        call_id = _to_text(entry.get("id") or entry.get("call_id"))
        tool_name = _to_text(entry.get("name") or entry.get("tool_name"))
        arguments = _normalize_tool_args(entry.get("arguments"))
        normalized.append(
            {
                "id": call_id,
                "name": tool_name,
                "arguments": arguments,
            },
        )
    return normalized


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "role": str(message.get("role", "")).strip().lower() or "assistant",
        "content": _to_text(message.get("content", "")),
        "tool_name": message.get("tool_name") if message.get("tool_name") is not None else None,
        "tool_call_id": message.get("tool_call_id") if message.get("tool_call_id") is not None else None,
        "tool_calls": _normalize_tool_calls(message.get("tool_calls")),
    }
    return normalized


def _find_last_matching_assistant(
    conversation: list[dict[str, Any]],
    expected_response: str,
) -> int:
    marker = _to_text(expected_response).strip()
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        if marker and _to_message_text(message) == marker:
            return index

    for index in range(len(conversation) - 1, -1, -1):
        if str(conversation[index].get("role", "")).strip().lower() == "assistant":
            return index
    return -1


def _find_user_boundary(
    conversation: list[dict[str, Any]],
    assistant_index: int,
    user_message: str,
) -> int:
    if assistant_index < 0:
        return 0

    marker = _to_text(user_message).strip()
    for index in range(assistant_index - 1, -1, -1):
        message = conversation[index]
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        if marker:
            text = _to_message_text(message)
            if text == marker:
                return index

    for index in range(assistant_index - 1, -1, -1):
        if str(conversation[index].get("role", "")).strip().lower() == "user":
            return index
    return max(assistant_index - 1, 0)


def assemble_round(
    *,
    user_message: str,
    assistant_response: str,
    conversation_history: object,
) -> list[dict[str, Any]]:
    conversation = _coerce_messages(conversation_history)
    if not conversation:
        return [
            {
                "role": "user",
                "content": _to_text(user_message),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": _to_text(assistant_response),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
        ]

    assistant_index = _find_last_matching_assistant(
        conversation=conversation,
        expected_response=assistant_response,
    )
    if assistant_index < 0:
        return [
            {
                "role": "user",
                "content": _to_text(user_message),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": _to_text(assistant_response),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
        ]

    user_index = _find_user_boundary(
        conversation=conversation,
        assistant_index=assistant_index,
        user_message=user_message,
    )

    selected = conversation[user_index : assistant_index + 1]
    if not selected:
        selected = [
            {
                "role": "user",
                "content": _to_text(user_message),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": _to_text(assistant_response),
                "tool_name": None,
                "tool_call_id": None,
                "tool_calls": [],
            },
        ]

    return [_normalize_message(message) for message in selected]


@dataclass(frozen=True, slots=True)
class RoundDigest:
    raw: str
    value: str


def compute_round_checksum(round_messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(
        round_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
