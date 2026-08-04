"""CoreGateway adapter backed by the versioned process IPC."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from .base import CoreGateway
from .contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    ContextViewResult,
    CoreCapabilityError,
    CoreGatewayError,
    CoreHealth,
    DomainRejectedError,
    RecordContextUsageCommand,
    RetrieveContextCommand,
    TransientCoreError,
)
from .maintenance import (
    BackupManifest,
    BeginRestoreCommand,
    BeginRestoreResult,
    CommitRestoreCommand,
    CommitRestoreResult,
    CreateBackupCommand,
    PrepareRestoreCommand,
    PrepareRestoreResult,
    RollbackRestoreCommand,
    RollbackRestoreResult,
    ValidateBackupCommand,
)
from .model_task_contracts import (
    CoreModelTask,
    FailModelTaskCommand,
    FailModelTaskResult,
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

_CAPABILITY_REQUIREMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "base": (
        frozenset(
            {
                "health",
                "accept_hypothesis",
                "retrieve_context",
                "record_context_usage",
            }
        ),
        frozenset(),
    ),
    "projections": (
        frozenset({"poll_projection_events", "ack_projection_events"}),
        frozenset({"projection_events"}),
    ),
    "model_tasks": (
        frozenset({"poll_model_tasks", "submit_model_result", "fail_model_task"}),
        frozenset({"model_task_failure_reporting"}),
    ),
    "maintenance": (
        frozenset(
            {
                "create_backup",
                "validate_backup",
                "prepare_restore",
                "begin_restore",
                "commit_restore",
                "rollback_restore",
            }
        ),
        frozenset({"core_owned_backup", "coordinated_restore"}),
    ),
}


class ProcessCoreGateway(CoreGateway):
    """Use a supervised Core process without importing the Python Core package."""

    def __init__(
        self,
        supervisor: CoreSupervisor,
        *,
        required_capabilities: Iterable[str] = (),
        required_operations: Iterable[str] = (),
    ) -> None:
        self._supervisor = supervisor
        self._required_capabilities = self._normalize_names(
            required_capabilities, "capability"
        )
        self._required_operations = self._normalize_names(
            required_operations, "operation"
        )
        if self._required_capabilities or self._required_operations:
            self._validate_capabilities()

    @staticmethod
    def _normalize_names(values: Iterable[str], label: str) -> tuple[str, ...]:
        names = tuple(values)
        if any(not isinstance(value, str) or not value.strip() for value in names):
            raise ValueError(f"{label} names must be non-empty strings")
        return tuple(dict.fromkeys(names))

    def require_capabilities(self, *capabilities: str) -> None:
        """Validate a feature at its consumer initialization boundary."""

        if len(capabilities) == 1 and not isinstance(capabilities[0], str):
            values = tuple(capabilities[0])
        else:
            values = capabilities
        names = self._normalize_names(values, "capability")
        current = getattr(self, "_required_capabilities", ())
        self._required_capabilities = tuple(dict.fromkeys((*current, *names)))
        self._validate_capabilities()

    def _fail_closed(self, error: CoreGatewayError) -> NoReturn:
        try:
            self._supervisor.close()
        finally:
            raise error

    def _validate_capabilities(self) -> None:
        requested = tuple(
            dict.fromkeys(
                (
                    *getattr(self, "_required_capabilities", ()),
                    *getattr(self, "_required_operations", ()),
                )
            )
        )
        required_operations = set(getattr(self, "_required_operations", ()))
        required_capability_flags: set[str] = set()
        for capability in getattr(self, "_required_capabilities", ()):
            try:
                operations, flags = _CAPABILITY_REQUIREMENTS[capability]
            except KeyError as exc:
                raise ValueError(f"unknown Core capability: {capability}") from exc
            required_operations.update(operations)
            required_capability_flags.update(flags)

        try:
            self._supervisor.start()
            handshake = self._supervisor.handshake_result
        except CoreGatewayError:
            raise
        except Exception as exc:
            raise TransientCoreError("Core capability handshake failed") from exc
        if not isinstance(handshake, Mapping):
            self._fail_closed(
                TransientCoreError("Core capability handshake is unavailable")
            )

        raw_operations = handshake.get("supported_operations", ())
        raw_flags = handshake.get("capabilities", {})
        if not isinstance(raw_operations, (list, tuple, set, frozenset)):
            self._fail_closed(
                TransientCoreError("Core capability operations are malformed")
            )
        if not isinstance(raw_flags, Mapping):
            self._fail_closed(TransientCoreError("Core capability flags are malformed"))
        advertised_operations = {
            operation
            for operation in raw_operations
            if isinstance(operation, str)
        }
        advertised_flags = {
            capability
            for capability, supported in raw_flags.items()
            if isinstance(capability, str) and supported is True
        }
        missing_operations = tuple(sorted(required_operations - advertised_operations))
        missing_flags = tuple(sorted(required_capability_flags - advertised_flags))
        if missing_operations or missing_flags:
            self._fail_closed(
                CoreCapabilityError(
                    requested=requested,
                    missing_operations=missing_operations,
                    missing_capabilities=missing_flags,
                )
            )

    @property
    def advertised_operations(self) -> frozenset[str]:
        """Return operations from the validated Core handshake."""

        handshake = self._supervisor.handshake_result
        if not isinstance(handshake, Mapping):
            return frozenset()
        raw = handshake.get("supported_operations", ())
        if not isinstance(raw, (list, tuple, set, frozenset)):
            return frozenset()
        return frozenset(item for item in raw if isinstance(item, str))

    @property
    def advertised_capabilities(self) -> frozenset[str]:
        """Return enabled feature flags from the Core handshake."""

        handshake = self._supervisor.handshake_result
        if not isinstance(handshake, Mapping):
            return frozenset()
        raw = handshake.get("capabilities", {})
        if not isinstance(raw, Mapping):
            return frozenset()
        return frozenset(
            capability
            for capability, supported in raw.items()
            if isinstance(capability, str) and supported is True
        )

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
        except (KeyError, TypeError, ValueError) as exc:
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

    def fail_model_task(
        self, command: FailModelTaskCommand
    ) -> FailModelTaskResult:
        result = self._request(
            "fail_model_task",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return FailModelTaskResult.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core model task failure result is malformed") from exc

    def create_backup(self, command: CreateBackupCommand) -> BackupManifest:
        result = self._request(
            "create_backup",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return BackupManifest.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core backup result is malformed") from exc

    def validate_backup(self, command: ValidateBackupCommand) -> BackupManifest:
        result = self._request(
            "validate_backup",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return BackupManifest.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core backup validation result is malformed") from exc

    def prepare_restore(
        self, command: PrepareRestoreCommand
    ) -> PrepareRestoreResult:
        result = self._request(
            "prepare_restore",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return PrepareRestoreResult.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core restore preparation result is malformed") from exc

    def begin_restore(self, command: BeginRestoreCommand) -> BeginRestoreResult:
        result = self._request(
            "begin_restore",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return BeginRestoreResult.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core restore begin result is malformed") from exc

    def commit_restore(self, command: CommitRestoreCommand) -> CommitRestoreResult:
        result = self._request(
            "commit_restore",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return CommitRestoreResult.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core restore commit result is malformed") from exc

    def rollback_restore(self, command: RollbackRestoreCommand) -> RollbackRestoreResult:
        result = self._request(
            "rollback_restore",
            command.to_payload(),
            request_id=command.request_id,
        )
        try:
            return RollbackRestoreResult.from_payload(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientCoreError("Core restore rollback result is malformed") from exc

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
