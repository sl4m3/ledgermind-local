from __future__ import annotations

import pytest

from ledgermind_local.core_gateway.model_task_contracts import (
    CoreModelTask,
    PollModelTasksCommand,
    SubmitModelResultCommand,
)
from ledgermind_local.core_gateway.process import ProcessCoreGateway


def _wire_task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "operation": "merge_knowledge",
        "memory_space_id": "space-1",
        "expected_versions": {"knowledge-a": 1, "knowledge-b": 2},
        "expires_at": "2026-08-03T12:00:00Z",
        "lease_expires_at": "2026-08-03T11:05:00Z",
        "model_input": {
            "items": [
                {
                    "reference": "knowledge-a",
                    "title": "A",
                    "target": "ops",
                    "statement": "Statement A",
                    "rationale": "Rationale A",
                    "required_constraints": ["keep source"],
                }
            ]
        },
    }


def test_model_task_wire_contract_is_strict_and_round_trips() -> None:
    task = CoreModelTask.from_wire(_wire_task())

    assert task.task_id == "task-1"
    assert task.operation == "merge_knowledge"
    assert task.to_payload()["expected_versions"] == {
        "knowledge-a": 1,
        "knowledge-b": 2,
    }

    invalid = dict(_wire_task(), internal_phase="pattern")
    with pytest.raises(ValueError, match="unknown fields"):
        CoreModelTask.from_wire(invalid)


def test_model_task_command_bounds_leases() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        PollModelTasksCommand("request-1", "space-1", "worker-1", limit=0)
    with pytest.raises(ValueError, match="between 1 and 3600"):
        PollModelTasksCommand("request-1", "space-1", "worker-1", lease_seconds=3601)


def test_process_gateway_decodes_model_task_poll_and_submit(monkeypatch) -> None:
    gateway = ProcessCoreGateway.__new__(ProcessCoreGateway)
    calls: list[tuple[str, dict[str, object], str]] = []

    def request(operation: str, payload: dict[str, object], *, request_id: str):
        calls.append((operation, payload, request_id))
        if operation == "poll_model_tasks":
            return {"tasks": [_wire_task()], "has_more": False}
        return {"accepted": True, "duplicate": True, "status": "completed"}

    monkeypatch.setattr(gateway, "_request", request)
    polled = gateway.poll_model_tasks(
        PollModelTasksCommand("poll-1", "space-1", "worker-1")
    )
    submitted = gateway.submit_model_result(
        SubmitModelResultCommand(
            request_id="submit-1",
            task_id="task-1",
            memory_space_id="space-1",
            worker_id="worker-1",
            result={"title": "Merged"},
        )
    )

    assert polled.tasks[0].task_id == "task-1"
    assert submitted.duplicate is True
    assert calls[0][0] == "poll_model_tasks"
    assert calls[1][0] == "submit_model_result"
