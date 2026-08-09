from __future__ import annotations

import pytest

from ledgermind_local.core_gateway.contracts import (
    CoreCapabilityError,
    TransientCoreError,
)
from ledgermind_local.core_gateway.process import ProcessCoreGateway

_OPERATIONS = [
    "handshake",
    "health",
    "ingest_raw_round",
    "poll_execution_tasks",
    "submit_execution_result",
    "fail_execution_task",
    "retrieve_context",
    "record_retrieval_outcome",
    "run_control_maintenance",
    "get_object_facet_statistics",
    "create_backup",
    "validate_backup",
    "prepare_restore",
    "begin_restore",
    "commit_restore",
    "rollback_restore",
    "shutdown",
]


class _Supervisor:
    def __init__(self, *, capabilities: dict[str, bool] | None = None) -> None:
        self.started = 0
        self.closed = 0
        self.handshake_result = {
            "supported_operations": list(_OPERATIONS),
            "capabilities": capabilities
            or {
            "core_owned_backup": True,
            "coordinated_restore": True,
            },
        }

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1


def test_process_gateway_validates_feature_capability_at_initialization() -> None:
    supervisor = _Supervisor()
    gateway = ProcessCoreGateway(
        supervisor,  # type: ignore[arg-type]
        required_capabilities=("maintenance",),
    )

    assert supervisor.started == 1
    assert "create_backup" in gateway.advertised_operations
    assert "core_owned_backup" in gateway.advertised_capabilities


def test_process_gateway_fails_closed_when_feature_flag_is_missing() -> None:
    supervisor = _Supervisor(
        capabilities={
            "core_owned_backup": False,
            "coordinated_restore": True,
        }
    )

    with pytest.raises(CoreCapabilityError) as error:
        ProcessCoreGateway(
            supervisor,  # type: ignore[arg-type]
            required_capabilities=("maintenance",),
        )

    assert error.value.missing_capabilities == ("core_owned_backup",)
    assert error.value.missing_operations == ()
    assert supervisor.closed == 1


def test_process_gateway_fails_closed_when_operation_is_missing() -> None:
    supervisor = _Supervisor()
    supervisor.handshake_result["supported_operations"] = [
        operation for operation in _OPERATIONS if operation != "prepare_restore"
    ]

    with pytest.raises(CoreCapabilityError) as error:
        ProcessCoreGateway(
            supervisor,  # type: ignore[arg-type]
            required_capabilities=("maintenance",),
        )

    assert error.value.missing_operations == ("prepare_restore",)
    assert supervisor.closed == 1


def test_process_gateway_closes_supervisor_on_malformed_handshake() -> None:
    supervisor = _Supervisor()
    supervisor.handshake_result["supported_operations"] = "malformed"

    with pytest.raises(TransientCoreError, match="operations are malformed"):
        ProcessCoreGateway(
            supervisor,  # type: ignore[arg-type]
            required_capabilities=("maintenance",),
        )

    assert supervisor.started == 1
    assert supervisor.closed == 1
