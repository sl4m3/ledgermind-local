from __future__ import annotations

import pytest

from ledgermind_local.core_gateway.contracts import (
    CoreExecutionResultV2,
    CoreExecutionTaskV2,
    RecordRetrievalOutcomeV2Command,
)

_EXPIRES_AT = "2026-08-08T12:00:00Z"


def _task_payload() -> dict[str, object]:
    return {
        "task_id": "task-v2",
        "task_kind": "generate_json",
        "operation": "core_owned_operation",
        "profile_slot": "operational",
        "memory_space_id": "space-v2",
        "expires_at": _EXPIRES_AT,
        "lease": "2026-08-08T11:05:00Z",
        "model_request": {
            "messages": [{"role": "user", "content": "opaque input"}],
            "max_output_tokens": 100,
            "response_format": "json_object",
        },
        "embedding_request": None,
        "operation_input": {"domain": {"owned_by": "core"}},
    }


def test_execution_task_round_trip_preserves_opaque_operation_metadata() -> None:
    task = CoreExecutionTaskV2.from_payload(_task_payload())

    assert task.task_kind == "generate_json"
    assert task.operation == "core_owned_operation"
    assert task.operation_input == {"domain": {"owned_by": "core"}}
    assert task.to_payload()["operation_input"] == {
        "domain": {"owned_by": "core"}
    }


def test_execution_result_rejects_unknown_fields_and_invalid_shape() -> None:
    payload = {
        "task_id": "task-v2",
        "task_kind": "generate_json",
        "status": "completed",
        "operation": "core_owned_operation",
        "operation_input": {},
        "output": {},
        "embedding_result": None,
        "egress_audit": {"status": "completed"},
        "error_code": None,
    }

    result = CoreExecutionResultV2.from_payload(payload)
    assert result.to_payload() == payload

    with pytest.raises(ValueError, match="unknown fields"):
        CoreExecutionResultV2.from_payload({**payload, "unexpected": True})
    with pytest.raises(ValueError, match="invalid output shape"):
        CoreExecutionResultV2.from_payload({**payload, "output": None})


def test_retrieval_outcome_requires_candidate_bound_delivery() -> None:
    with pytest.raises(ValueError, match="subset"):
        RecordRetrievalOutcomeV2Command(
            request_id="outcome-request",
            retrieval_request_id="retrieval-request",
            candidate_value_ids=("candidate-1",),
            delivered_value_ids=("other-value",),
        )
