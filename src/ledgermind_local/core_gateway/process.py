"""CoreGateway adapter backed by the versioned process IPC."""

from __future__ import annotations

import json
from typing import Any

from .base import CoreGateway
from .contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    ContextViewResult,
    CoreGatewayError,
    CoreHealth,
    DomainRejectedError,
    RecordContextUsageCommand,
    RetrieveContextCommand,
    TransientCoreError,
)
from .model_task_contracts import (
    CoreModelTask,
    PollModelTasksCommand,
    PollModelTasksResult,
    SubmitModelResult,
    SubmitModelResultCommand,
)
from .projection_contracts import (
    AckProjectionEventsCommand,
    AckProjectionEventsResult,
    CoreProjectionEvent,
    PollProjectionEventsCommand,
    PollProjectionEventsResult,
)
from .supervisor import (
    CoreSupervisor,
    CoreSupervisorError,
    CoreSupervisorRemoteError,
)


class ProcessCoreGateway(CoreGateway):
    """Use a supervised Core process without importing the Python Core package."""

    def __init__(self, supervisor: CoreSupervisor) -> None:
        self._supervisor = supervisor

    def health(self) -> CoreHealth:
        try:
            result = self._supervisor.request("health", {})
        except CoreSupervisorError as exc:
            return CoreHealth(healthy=False, backend="process", detail=str(exc))
        return CoreHealth(
            healthy=bool(result.get("healthy", False)),
            backend="process",
            detail=str(result["detail"]) if result.get("detail") else None,
        )

    def accept_hypothesis(
        self, command: AcceptHypothesisCommand
    ) -> AcceptHypothesisResult:
        result = self._request(
            "accept_hypothesis",
            command.to_payload(),
            request_id=command.command_id,
        )
        return AcceptHypothesisResult(
            accepted=bool(result.get("accepted", False)),
            duplicate=bool(result.get("duplicate", False)),
            core_reference_id=(
                str(result["core_reference_id"])
                if result.get("core_reference_id") is not None
                else None
            ),
            result_json=(
                str(result["result_json"])
                if result.get("result_json") is not None
                else None
            ),
        )

    def retrieve_context(self, request: RetrieveContextCommand) -> ContextViewResult:
        payload: dict[str, object] = {
            "memory_space_id": request.memory_space_id,
            "query": request.query,
            "limit": request.limit,
        }
        if request.candidate_ids:
            payload["candidate_ids"] = list(request.candidate_ids)
            payload["candidate_scores"] = [
                {"knowledge_id": knowledge_id, "score": score}
                for knowledge_id, score in request.candidate_scores
            ]
        result = self._request(
            "retrieve_context",
            payload,
            request_id=request.request_id,
        )
        return ContextViewResult.from_core_json(
            json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        )

    def record_context_usage(self, command: RecordContextUsageCommand) -> None:
        self._request(
            "record_context_usage",
            {
                "memory_space_id": command.memory_space_id,
                "item_ids": list(command.item_ids),
            },
            request_id=command.request_id,
        )

    def poll_projection_events(
        self, command: PollProjectionEventsCommand
    ) -> PollProjectionEventsResult:
        result = self._request(
            "poll_projection_events",
            command.to_payload(),
            request_id=command.request_id,
        )
        raw_events = result.get("events", [])
        if not isinstance(raw_events, list):
            raise TransientCoreError("Core projection event result is malformed")
        try:
            events = tuple(CoreProjectionEvent.from_wire(dict(item)) for item in raw_events)
        except (TypeError, ValueError) as exc:
            raise DomainRejectedError("invalid_projection_event", str(exc)) from exc
        return PollProjectionEventsResult(
            events=events,
            has_more=bool(result.get("has_more", False)),
        )

    def ack_projection_events(
        self, command: AckProjectionEventsCommand
    ) -> AckProjectionEventsResult:
        result = self._request(
            "ack_projection_events",
            command.to_payload(),
            request_id=command.request_id,
        )
        acknowledged = result.get("acknowledged", [])
        if not isinstance(acknowledged, list) or any(
            not isinstance(event_id, str) or not event_id for event_id in acknowledged
        ):
            raise TransientCoreError("Core projection acknowledgement is malformed")
        return AckProjectionEventsResult(acknowledged=tuple(acknowledged))

    def poll_model_tasks(
        self, command: PollModelTasksCommand
    ) -> PollModelTasksResult:
        result = self._request(
            "poll_model_tasks",
            command.to_payload(),
            request_id=command.request_id,
        )
        raw_tasks = result.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise TransientCoreError("Core model task result is malformed")
        try:
            tasks = tuple(CoreModelTask.from_wire(dict(item)) for item in raw_tasks)
        except (TypeError, ValueError) as exc:
            raise DomainRejectedError("invalid_model_task", str(exc)) from exc
        return PollModelTasksResult(
            tasks=tasks,
            has_more=bool(result.get("has_more", False)),
        )

    def submit_model_result(
        self, command: SubmitModelResultCommand
    ) -> SubmitModelResult:
        result = self._request(
            "submit_model_result",
            command.to_payload(),
            request_id=command.request_id,
        )
        accepted = result.get("accepted")
        duplicate = result.get("duplicate")
        status = result.get("status")
        if (
            not isinstance(accepted, bool)
            or not isinstance(duplicate, bool)
            or not isinstance(status, str)
            or not status.strip()
        ):
            raise TransientCoreError("Core model result acknowledgement is malformed")
        return SubmitModelResult(
            accepted=accepted,
            duplicate=duplicate,
            status=status,
        )

    def close(self) -> None:
        self._supervisor.close()

    def _request(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        try:
            return self._supervisor.request(
                operation,
                payload,
                request_id=request_id,
            )
        except CoreSupervisorRemoteError as exc:
            if exc.retryable:
                raise TransientCoreError(str(exc)) from exc
            raise DomainRejectedError(exc.code, exc.detail) from exc
        except CoreSupervisorError as exc:
            raise TransientCoreError(str(exc)) from exc
        except CoreGatewayError:
            raise
        except Exception as exc:
            raise TransientCoreError("Core process request failed") from exc


__all__ = ["ProcessCoreGateway"]
