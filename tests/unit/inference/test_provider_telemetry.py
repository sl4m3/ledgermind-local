from __future__ import annotations

import json

from ledgermind_local.inference.provider_telemetry import record_http_attempt


def test_http_telemetry_has_correlated_accounting_fields(tmp_path, monkeypatch) -> None:
    target = tmp_path / "provider-telemetry.jsonl"
    monkeypatch.setenv("LEDGERMIND_PROVIDER_TELEMETRY_PATH", str(target))

    record_http_attempt(
        kind="generation",
        operation="capability_probe",
        provider_profile_fingerprint="sha256:" + "a" * 64,
        transport="openai_compatible",
        model="test-model",
        duration_ms=1.25,
        status="completed",
        task_id="task-1",
        root_task_id="root-1",
        attempt_index=2,
        retry_index=1,
        input_tokens=4,
        output_tokens=3,
        operation_item_counts={"knowledge_values": 2, "facets": 1},
    )

    event = json.loads(target.read_text(encoding="utf-8"))
    assert event["request_id"].startswith("local-")
    assert event["task_id"] == "task-1"
    assert event["root_task_id"] == "root-1"
    assert event["attempt_index"] == 2
    assert event["request_reason"] == "transport_retry"
    assert event["input_tokens"] == 4
    assert event["output_tokens"] == 3
    assert event["total_tokens"] == 7
    assert event["usage_unknown"] is False
    assert event["operation_item_counts"] == {"facets": 1, "knowledge_values": 2}


def test_probe_without_usage_is_explicitly_unknown(tmp_path, monkeypatch) -> None:
    target = tmp_path / "provider-telemetry.jsonl"
    monkeypatch.setenv("LEDGERMIND_PROVIDER_TELEMETRY_PATH", str(target))

    record_http_attempt(
        kind="generation",
        operation="capability_probe",
        transport="openai_compatible",
        model="test-model",
        duration_ms=0,
        status="completed",
    )

    event = json.loads(target.read_text(encoding="utf-8"))
    assert event["request_reason"] == "provider_probe"
    assert event["usage_unknown"] is True
    assert event["input_tokens"] is None
    assert event["output_tokens"] is None
    assert event["total_tokens"] is None


def test_http_attempt_records_safe_provider_routing(tmp_path, monkeypatch) -> None:
    target = tmp_path / "provider-telemetry.jsonl"
    monkeypatch.setenv("LEDGERMIND_PROVIDER_TELEMETRY_PATH", str(target))

    record_http_attempt(
        kind="generation",
        operation="execution_semantic",
        transport="openai_compatible",
        model="test-model",
        duration_ms=12.5,
        status="invalid_response",
        configured_routes=("baidu/fp8", "deepinfra/fp8"),
        served_by="Baidu",
    )

    event = json.loads(target.read_text(encoding="utf-8"))
    assert event["configured_routes"] == ["baidu/fp8", "deepinfra/fp8"]
    assert event["served_by"] == "Baidu"
