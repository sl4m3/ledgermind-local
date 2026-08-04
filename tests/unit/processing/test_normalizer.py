from __future__ import annotations

import copy

import pytest

from ledgermind_local.processing.normalizer import normalize_raw_round


def test_normalizer_redacts_secret_like_values_and_is_deterministic() -> None:
    payload = {
        "source": {
            "system": "hermes",
            "instance_id": "instance-1",
            "profile_id": "default",
            "session_id": "session-1",
            "round_id": "round-1",
        },
        "round": {
            "started_at": "2026-08-02T20:00:00Z",
            "completed_at": "2026-08-02T20:01:00Z",
            "events": [
                {
                    "event_id": "m-1",
                    "sequence": 0,
                    "kind": "message",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Use token=super-secret to deploy"},
                    ],
                },
                {
                    "event_id": "t-1",
                    "sequence": 1,
                    "kind": "tool_call",
                    "tool_call_id": "call-1",
                    "tool_name": "deploy",
                    "arguments": {
                        "api_key": "secret-key-value",
                        "environment": "staging",
                    },
                },
                {
                    "event_id": "r-1",
                    "sequence": 2,
                    "kind": "tool_result",
                    "tool_call_id": "call-1",
                    "status": "success",
                    "content": [{"type": "text", "text": "Deployed to staging"}],
                },
                {
                    "event_id": "m-2",
                    "sequence": 3,
                    "kind": "message",
                    "role": "assistant",
                    "final": True,
                    "content": [
                        {"type": "text", "text": "Deployed to staging"},
                    ],
                },
            ],
        },
    }

    first = normalize_raw_round(payload)
    second = normalize_raw_round(payload)

    assert first == second
    assert first.user_text == "Use token=[REDACTED] to deploy"
    assert first.transcript == (
        "user: Use token=[REDACTED] to deploy\n"
        'tool:deploy args={"api_key":"[REDACTED]","environment":"staging"}\n'
        "tool_result:call-1 status=success result=Deployed to staging\n"
        "assistant: Deployed to staging"
    )
    assert first.assistant_text == "Deployed to staging"
    interaction = first.tool_interactions[0]
    assert interaction.tool_call_id == "call-1"
    assert interaction.source_call_event_id == "t-1"
    assert interaction.source_result_event_id == "r-1"
    assert interaction.result_text == "Deployed to staging"
    assert interaction.result_json is None
    assert interaction.status == "success"
    assert interaction.error_text == ""
    assert "super-secret" not in first.transcript
    assert "secret-key-value" not in first.transcript
    assert first.normalized_digest.startswith("sha256:")
    assert len(first.normalized_digest) == 71

    changed = copy.deepcopy(payload)
    changed["round"]["events"][2]["content"][0]["text"] = "Deployment failed"
    assert normalize_raw_round(changed).normalized_digest != first.normalized_digest


def test_normalizer_keeps_missing_tool_result_as_unknown() -> None:
    payload = {
        "memory_space_id": "space",
        "source": {
            "system": "hermes",
            "instance_id": "instance",
            "profile_id": "profile",
            "session_id": "session",
            "round_id": "round",
        },
        "round": {
            "started_at": "2026-08-02T20:00:00Z",
            "completed_at": "2026-08-02T20:01:00Z",
            "events": [
                {
                    "event_id": "call-event",
                    "sequence": 0,
                    "kind": "tool_call",
                    "tool_call_id": "call-unknown",
                    "tool_name": "inspect",
                    "arguments": {"path": "README.md"},
                }
            ],
        },
    }

    interaction = normalize_raw_round(payload).tool_interactions[0]

    assert interaction.status == "unknown"
    assert interaction.source_result_event_id is None
    assert interaction.result_text == ""
    assert interaction.error_text == ""


def test_normalizer_preserves_tool_error_and_json_result() -> None:
    payload = {
        "memory_space_id": "space",
        "source": {"system": "hermes"},
        "round": {
            "started_at": "2026-08-02T20:00:00Z",
            "completed_at": "2026-08-02T20:01:00Z",
            "events": [
                {
                    "event_id": "call-event",
                    "sequence": 0,
                    "kind": "tool_call",
                    "tool_call_id": "call-error",
                    "tool_name": "deploy",
                    "arguments": {},
                },
                {
                    "event_id": "result-event",
                    "sequence": 1,
                    "kind": "tool_result",
                    "tool_call_id": "call-error",
                    "status": "error",
                    "error": "token=secret-value rejected",
                    "content": [
                        {"type": "text", "text": "deployment failed"},
                        {"type": "json", "data": {"attempts": 1}},
                    ],
                },
            ],
        },
    }

    interaction = normalize_raw_round(payload).tool_interactions[0]

    assert interaction.status == "error"
    assert interaction.result_text == "deployment failed"
    assert interaction.result_json == '{"attempts":1}'
    assert interaction.error_text == "token=[REDACTED] rejected"


def test_normalizer_rejects_tool_result_without_call() -> None:
    payload = {
        "memory_space_id": "space",
        "source": {"system": "hermes"},
        "round": {
            "started_at": "2026-08-02T20:00:00Z",
            "completed_at": "2026-08-02T20:01:00Z",
            "events": [
                {
                    "event_id": "result-event",
                    "sequence": 0,
                    "kind": "tool_result",
                    "tool_call_id": "missing-call",
                    "status": "error",
                    "content": [{"type": "text", "text": "not found"}],
                }
            ],
        },
    }

    with pytest.raises(ValueError, match="without a matching tool_call"):
        normalize_raw_round(payload)
