from __future__ import annotations

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
                    "event_id": "m-2",
                    "sequence": 2,
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
        'user: Use token=[REDACTED] to deploy\n'
        'tool:deploy args={"api_key":"[REDACTED]","environment":"staging"}\n'
        "assistant: Deployed to staging"
    )
    assert first.assistant_text == "Deployed to staging"
    assert first.tool_calls[0].arguments_json == '{"api_key":"[REDACTED]","environment":"staging"}'
    assert "super-secret" not in first.transcript
    assert "secret-key-value" not in first.transcript
    assert first.normalized_digest.startswith("sha256:")
    assert len(first.normalized_digest) == 71
