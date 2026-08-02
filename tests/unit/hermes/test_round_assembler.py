"""Tests for Hermes round assembly helpers."""

from __future__ import annotations

import hashlib
import json

from ledgermind_local.plugins.hermes.round_assembler import (
    assemble_round,
    compute_round_checksum,
)


def test_round_checksum_matches_full_serialized_messages() -> None:
    messages = [
        {
            "role": "user",
            "content": "вопрос",
            "tool_name": None,
            "tool_call_id": None,
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "ответ",
            "tool_name": None,
            "tool_call_id": None,
            "tool_calls": [],
        },
    ]

    checksum = compute_round_checksum(messages)
    expected = "sha256:" + hashlib.sha256(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()
    assert checksum == expected


def test_assemble_round_prefers_latest_matching_turn() -> None:
    conversation_history = [
        {"role": "user", "content": "старый запрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "повторный запрос"},
        {"role": "assistant", "content": "старый ответ"},
        {
            "role": "user",
            "content": "текущий запрос",
            "tool_calls": [{"name": "calc", "arguments": "{}", "id": "tool-1"}],
        },
        {
            "role": "assistant",
            "content": "последний ответ",
            "tool_call_id": "tool-1",
        },
    ]

    result = assemble_round(
        user_message="текущий запрос",
        assistant_response="последний ответ",
        conversation_history=conversation_history,
    )

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "текущий запрос"
    assert result[1]["content"] == "последний ответ"


def test_assemble_round_falls_back_to_last_assistant_without_exact_match() -> None:
    conversation_history = [
        {"role": "user", "content": "вопрос"},
        {
            "role": "assistant",
            "content": "ответ",
            "tool_calls": [{"name": "a", "arguments": "{}", "id": "id"}],
        },
        {"role": "assistant", "content": "повторный ответ"},
    ]

    result = assemble_round(
        user_message="вопрос",
        assistant_response="не совпадает",
        conversation_history=conversation_history,
    )

    assert len(result) == 3
    assert result[0]["content"] == "вопрос"
    assert result[1]["content"] == "ответ"
    assert result[1]["tool_calls"] == [{"id": "id", "name": "a", "arguments": "{}"}]
    assert result[2]["content"] == "повторный ответ"
    assert result[2]["tool_calls"] == []
    assert len(compute_round_checksum(result)) == 71
