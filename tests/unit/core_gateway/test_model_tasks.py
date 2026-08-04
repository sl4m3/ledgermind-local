from __future__ import annotations

import pytest

from ledgermind_local.core_gateway.contracts import DomainRejectedError
from ledgermind_local.core_gateway.isolation import IsolationCapabilities
from ledgermind_local.core_gateway.maintenance import (
    BackupManifest,
    CoreMaintenanceRunner,
    CoreRestoreError,
    CreateBackupCommand,
    PrepareRestoreCommand,
    PrepareRestoreResult,
    ValidateBackupCommand,
)
from ledgermind_local.core_gateway.model_task_contracts import (
    CoreModelTask,
    FailModelTaskCommand,
    FailModelTaskResult,
    PollModelTasksCommand,
    SubmitModelResultCommand,
)
from ledgermind_local.core_gateway.process import ProcessCoreGateway
from ledgermind_local.core_gateway.sandbox import SandboxPlan
from ledgermind_local.core_gateway.signing import CoreBinaryVerification


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


def test_process_gateway_rejects_incomplete_model_task_payload(monkeypatch) -> None:
    gateway = ProcessCoreGateway.__new__(ProcessCoreGateway)
    monkeypatch.setattr(
        gateway,
        "_request",
        lambda *args, **kwargs: {"tasks": [{}], "has_more": False},
    )

    with pytest.raises(DomainRejectedError, match="invalid_model_task"):
        gateway.poll_model_tasks(
            PollModelTasksCommand("poll-1", "space-1", "worker-1")
        )


def test_failure_and_backup_commands_round_trip_strict_payloads() -> None:
    failure = FailModelTaskCommand(
        request_id="fail-1",
        task_id="task-1",
        memory_space_id="space-1",
        worker_id="worker-1",
        error_code="provider_timeout",
        retryable=True,
        retry_after_seconds=30,
        failed_at="2026-08-04T12:00:00Z",
    )
    assert failure.to_payload() == {
        "task_id": "task-1",
        "memory_space_id": "space-1",
        "worker_id": "worker-1",
        "error_code": "provider_timeout",
        "retryable": True,
        "retry_after_seconds": 30,
        "failed_at": "2026-08-04T12:00:00Z",
    }
    result = FailModelTaskResult.from_payload(
        {
            "status": "pending",
            "attempts": 2,
            "available_at": "2026-08-04T12:00:30Z",
            "last_error_code": "provider_timeout",
            "failed_at": "2026-08-04T12:00:00Z",
            "completed_at": None,
        }
    )
    assert result.status == "pending"
    assert result.attempts == 2

    manifest = BackupManifest.from_payload(
        {
            "relative_path": "exchange/outgoing/backup-1.sqlite",
            "sha256": "sha256:" + "a" * 64,
            "size_bytes": 12,
            "schema_version": 6,
        }
    )
    assert manifest.to_payload()["schema_version"] == 6
    assert CreateBackupCommand("backup-1").to_payload() == {}
    assert ValidateBackupCommand(
        "validate-1", "exchange/incoming/backup-1.sqlite", manifest.sha256
    ).to_payload()["relative_path"] == "exchange/incoming/backup-1.sqlite"
    assert PrepareRestoreCommand(
        "prepare-1", "exchange/incoming/backup-1.sqlite", manifest.sha256
    ).to_payload()["sha256"] == manifest.sha256


def test_process_gateway_decodes_failure_and_backup_operations(monkeypatch) -> None:
    gateway = ProcessCoreGateway.__new__(ProcessCoreGateway)
    calls: list[tuple[str, dict[str, object], str]] = []

    def request(operation: str, payload: dict[str, object], *, request_id: str):
        calls.append((operation, payload, request_id))
        if operation == "fail_model_task":
            return {
                "status": "failed",
                "attempts": 2,
                "available_at": None,
                "last_error_code": "invalid_model_output",
                "failed_at": "2026-08-04T12:00:00Z",
                "completed_at": "2026-08-04T12:00:00Z",
            }
        if operation == "create_backup":
            return {
                "relative_path": "exchange/outgoing/backup-1.sqlite",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": 12,
                "schema_version": 6,
            }
        if operation == "prepare_restore":
            return {
                "relative_path": "exchange/incoming/backup-1.sqlite",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": 12,
                "schema_version": 6,
                "restore_token": "restore-token-1",
                "requires_restart": True,
            }
        return {
            "relative_path": "exchange/incoming/backup-1.sqlite",
            "sha256": "sha256:" + "a" * 64,
            "size_bytes": 12,
            "schema_version": 6,
        }

    monkeypatch.setattr(gateway, "_request", request)
    failed = gateway.fail_model_task(
        FailModelTaskCommand(
            request_id="fail-1",
            task_id="task-1",
            memory_space_id="space-1",
            worker_id="worker-1",
            error_code="invalid_model_output",
            retryable=False,
            retry_after_seconds=0,
            failed_at="2026-08-04T12:00:00Z",
        )
    )
    created = gateway.create_backup(CreateBackupCommand("backup-1"))
    validated = gateway.validate_backup(
        ValidateBackupCommand(
            "validate-1", "exchange/incoming/backup-1.sqlite", created.sha256
        )
    )
    prepared = gateway.prepare_restore(
        PrepareRestoreCommand(
            "prepare-1", validated.relative_path, validated.sha256
        )
    )

    assert failed.status == "failed"
    assert isinstance(created, BackupManifest)
    assert validated.relative_path.startswith("exchange/incoming/")
    assert isinstance(prepared, PrepareRestoreResult)
    assert prepared.restore_token == "restore-token-1"
    assert [call[0] for call in calls] == [
        "fail_model_task",
        "create_backup",
        "validate_backup",
        "prepare_restore",
    ]


def test_core_maintenance_runner_uses_sandbox_and_never_receives_working_db(
    tmp_path,
) -> None:
    import subprocess

    calls: list[str] = []
    command_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    preparation = PrepareRestoreResult(
        relative_path="exchange/incoming/restore.bin",
        sha256="sha256:" + "a" * 64,
        size_bytes=1,
        schema_version=6,
        restore_token="restore-token",
        requires_restart=True,
    )

    def verify(path, **kwargs):
        return CoreBinaryVerification(
            binary_path=path,
            sha256="a" * 64,
            signature_path=tmp_path / "signature",
            public_key_path=tmp_path / "public-key",
        )

    def build(command, **kwargs):
        command_calls.append((tuple(command), kwargs))
        return SandboxPlan(
            command=("sandbox", *command),
            capabilities=IsolationCapabilities(rounds_database_hidden=True),
        )

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = CoreMaintenanceRunner(
        core_binary_path=tmp_path / "core",
        core_data_dir=tmp_path / "core-data",
        rounds_database_path=tmp_path / "rounds.db",
        stop_gateway=lambda: calls.append("stop"),
        start_gateway=lambda: calls.append("start"),
        health_check=lambda: type("Health", (), {"healthy": True})(),
        binary_verifier=verify,
        sandbox_builder=build,
        command_runner=run,
    )

    runner.apply_restore(preparation)

    assert calls == ["stop", "start"]
    command = command_calls[0][0]
    assert "--apply-restore" in command
    assert "--restore-token" in command
    assert all("knowledge.db" not in part for part in command)
    assert command_calls[0][1]["strict"] is True


def test_core_maintenance_runner_restarts_gateway_after_restore_failure(tmp_path) -> None:
    import subprocess

    calls: list[str] = []
    preparation = PrepareRestoreResult(
        relative_path="exchange/incoming/restore.bin",
        sha256="sha256:" + "a" * 64,
        size_bytes=1,
        schema_version=6,
        restore_token="restore-token",
        requires_restart=True,
    )

    def verify(path, **kwargs):
        return CoreBinaryVerification(
            binary_path=path,
            sha256="a" * 64,
            signature_path=tmp_path / "signature",
            public_key_path=tmp_path / "public-key",
        )

    runner = CoreMaintenanceRunner(
        core_binary_path=tmp_path / "core",
        core_data_dir=tmp_path / "core-data",
        stop_gateway=lambda: calls.append("stop"),
        start_gateway=lambda: calls.append("start"),
        health_check=lambda: {"healthy": True},
        binary_verifier=verify,
        sandbox_builder=lambda command, **kwargs: SandboxPlan(
            command=tuple(command), capabilities=IsolationCapabilities()
        ),
        command_runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 17, stdout="", stderr="restore failed"
        ),
    )

    with pytest.raises(CoreRestoreError, match="restore command failed"):
        runner.apply_restore(preparation)
    assert calls == ["stop", "start"]


def test_core_maintenance_runner_rejects_unverified_binary(tmp_path) -> None:
    preparation = PrepareRestoreResult(
        relative_path="exchange/incoming/restore.bin",
        sha256="sha256:" + "a" * 64,
        size_bytes=1,
        schema_version=6,
        restore_token="restore-token",
        requires_restart=True,
    )
    sandbox_calls: list[object] = []

    def verify(path, **kwargs):
        return CoreBinaryVerification(
            binary_path=path,
            sha256="a" * 64,
            signature_path=tmp_path / "signature",
            public_key_path=tmp_path / "public-key",
            signature_verified=False,
        )

    def build(command, **kwargs):
        sandbox_calls.append(command)
        raise AssertionError("sandbox must not run")

    runner = CoreMaintenanceRunner(
        core_binary_path=tmp_path / "core",
        core_data_dir=tmp_path / "core-data",
        stop_gateway=lambda: None,
        start_gateway=lambda: None,
        health_check=lambda: {"healthy": True},
        binary_verifier=verify,
        sandbox_builder=build,
    )

    with pytest.raises(CoreRestoreError, match="signature"):
        runner.apply_restore(preparation)
    assert sandbox_calls == []


def test_core_maintenance_runner_wraps_restart_failure(tmp_path) -> None:
    import subprocess

    preparation = PrepareRestoreResult(
        relative_path="exchange/incoming/restore.bin",
        sha256="sha256:" + "a" * 64,
        size_bytes=1,
        schema_version=6,
        restore_token="restore-token",
        requires_restart=True,
    )
    calls: list[str] = []

    def verify(path, **kwargs):
        return CoreBinaryVerification(
            binary_path=path,
            sha256="a" * 64,
            signature_path=tmp_path / "signature",
            public_key_path=tmp_path / "public-key",
        )

    def start() -> None:
        calls.append("start")
        raise RuntimeError("restart exploded")

    runner = CoreMaintenanceRunner(
        core_binary_path=tmp_path / "core",
        core_data_dir=tmp_path / "core-data",
        stop_gateway=lambda: calls.append("stop"),
        start_gateway=start,
        health_check=lambda: {"healthy": True},
        binary_verifier=verify,
        sandbox_builder=lambda command, **kwargs: SandboxPlan(
            command=tuple(command), capabilities=IsolationCapabilities()
        ),
        command_runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    with pytest.raises(CoreRestoreError, match="restart"):
        runner.apply_restore(preparation)
    assert calls == ["stop", "start"]
