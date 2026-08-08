from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import ledgermind_protocol
import pydantic_core
import pytest

from ledgermind_local.core_gateway.contracts import (
    AcceptHypothesisCommand,
    CoreCapabilityError,
    DomainRejectedError,
    FailExecutionTaskCommand,
    HypothesisEvidence,
    HypothesisExtraction,
    HypothesisPayload,
    IngestRawRoundCommand,
    PollExecutionTasksCommand,
    RecordContextUsageCommand,
    RecordRetrievalOutcomeV2Command,
    RetrieveContextCommand,
    RetrieveContextV2Command,
    SubmitExecutionResultCommand,
)
from ledgermind_local.core_gateway.maintenance import (
    CreateBackupCommand,
    PrepareRestoreCommand,
    ValidateBackupCommand,
)
from ledgermind_local.core_gateway.model_task_contracts import FailModelTaskCommand
from ledgermind_local.core_gateway.process import ProcessCoreGateway
from ledgermind_local.core_gateway.supervisor import CoreSupervisor as _CoreSupervisor

_FAKE_PROCESS = (
    Path(__file__).resolve().parents[2] / "fixtures" / "fake_core_process.py"
)
_LOCAL_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_SOURCE = _LOCAL_ROOT / "src"
_PROTOCOL_SOURCE = Path(ledgermind_protocol.__file__).resolve().parent.parent
_SITE_PACKAGES = Path(sysconfig.get_path("purelib"))
_PYDANTIC_CORE = next(
    Path(pydantic_core.__file__).resolve().parent.glob("*.so")
)
_RUNTIME_PATHS = (_LOCAL_SOURCE, _PROTOCOL_SOURCE, _SITE_PACKAGES, _PYDANTIC_CORE)


def _command(*args: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(_FAKE_PROCESS),
        "--python-path",
        str(_LOCAL_SOURCE),
        "--python-path",
        str(_PROTOCOL_SOURCE),
        "--python-path",
        str(_SITE_PACKAGES),
        *args,
    )


def _test_supervisor(command: tuple[str, ...], **kwargs: Any) -> _CoreSupervisor:
    """Declare the sibling protocol source needed by the isolated fixture."""

    return _CoreSupervisor(
        command,
        runtime_paths=_RUNTIME_PATHS,
        **kwargs,
    )


def _accept_command(statement: str = "same statement") -> AcceptHypothesisCommand:
    digest = "sha256:" + "a" * 64
    return AcceptHypothesisCommand(
        protocol_version=1,
        command_id="command-1",
        idempotency_key=digest,
        memory_space_id="space-1",
        hypothesis=HypothesisPayload(
            hypothesis_id="hypothesis-1",
            content_digest=digest,
            title="Title",
            target="Target",
            statement=statement,
            rationale="Rationale",
            result="Result",
            artifacts=(),
            evidence=HypothesisEvidence(
                source_system="tests",
                source_instance_id="instance-1",
                source_profile_id="profile-1",
                source_session_id="session-1",
                source_round_id="round-1",
                raw_round_digest=digest,
                normalized_round_digest=digest,
                source_event_ids=("event-1",),
            ),
            extraction=HypothesisExtraction(
                provider="test",
                model="fake",
                prompt_version=1,
                schema_version=1,
                completed_at="2026-08-03T00:00:00Z",
            ),
        ),
    )


def test_process_gateway_performs_handshake_and_health() -> None:
    supervisor = _test_supervisor(_command(), startup_timeout_seconds=2.0)
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
    finally:
        gateway.close()

    assert health.healthy is True
    assert health.backend == "process"


def test_process_gateway_restarts_after_child_crash(tmp_path: Path) -> None:
    crash_marker = tmp_path / "crash-once.marker"
    supervisor = _test_supervisor(
        _command("--crash-once-file", str(crash_marker)),
        startup_timeout_seconds=2.0,
        core_data_dir=tmp_path,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        first_health = gateway.health()
        second_health = gateway.health()
    finally:
        gateway.close()

    assert first_health.healthy is False
    assert second_health.healthy is True


def test_process_gateway_marks_timeout_unhealthy_and_terminates_child() -> None:
    supervisor = _test_supervisor(
        _command("--delay-seconds", "0.2"),
        startup_timeout_seconds=2.0,
        operation_timeout_seconds=0.05,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
        assert supervisor.pid is None
    finally:
        gateway.close()

    assert health.healthy is False
    assert "timed out" in (health.detail or "")


def test_accept_hypothesis_replay_is_idempotent() -> None:
    supervisor = _test_supervisor(_command(), startup_timeout_seconds=2.0)
    gateway = ProcessCoreGateway(supervisor)
    command = _accept_command()

    try:
        first = gateway.accept_hypothesis(command)
        replay = gateway.accept_hypothesis(command)
    finally:
        gateway.close()

    assert first.accepted is True
    assert first.duplicate is False
    assert replay.accepted is True
    assert replay.duplicate is True


def test_accept_hypothesis_idempotency_conflict_is_domain_rejection() -> None:
    supervisor = _test_supervisor(_command(), startup_timeout_seconds=2.0)
    gateway = ProcessCoreGateway(supervisor)

    try:
        gateway.accept_hypothesis(_accept_command("first statement"))
        with pytest.raises(DomainRejectedError) as error:
            gateway.accept_hypothesis(_accept_command("different statement"))
    finally:
        gateway.close()

    assert error.value.code == "IDEMPOTENCY_CONFLICT"


def test_process_gateway_retrieves_context_and_records_usage() -> None:
    supervisor = _test_supervisor(_command(), startup_timeout_seconds=2.0)
    gateway = ProcessCoreGateway(supervisor)

    try:
        context = gateway.retrieve_context(
            RetrieveContextCommand(
                request_id="retrieve-1",
                memory_space_id="space-1",
                query="query",
                limit=3,
            )
        )
        gateway.record_context_usage(
            RecordContextUsageCommand(
                request_id="usage-1",
                memory_space_id="space-1",
                item_ids=("knowledge-1",),
            )
        )
    finally:
        gateway.close()

    assert context.api_version == "1"
    assert len(context.items) == 1
    assert context.items[0].knowledge_id == "knowledge-1"


def test_process_gateway_consumes_model_failure_and_core_backup_operations(
    tmp_path: Path,
) -> None:
    supervisor = _test_supervisor(
        _command(), startup_timeout_seconds=2.0, core_data_dir=tmp_path
    )
    gateway = ProcessCoreGateway(supervisor, required_capabilities=("maintenance",))

    try:
        failure = gateway.fail_model_task(
            FailModelTaskCommand(
                request_id="failure-1",
                task_id="task-1",
                memory_space_id="space-1",
                worker_id="worker-1",
                error_code="provider_timeout",
                retryable=True,
                retry_after_seconds=30,
                failed_at="2026-08-04T12:00:00Z",
            )
        )
        created = gateway.create_backup(CreateBackupCommand("backup-1"))
        outgoing = tmp_path / created.relative_path
        incoming = tmp_path / "exchange" / "incoming" / outgoing.name
        incoming.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outgoing, incoming)
        validated = gateway.validate_backup(
            ValidateBackupCommand("validate-1", f"exchange/incoming/{outgoing.name}", created.sha256)
        )
        prepared = gateway.prepare_restore(
            PrepareRestoreCommand("prepare-1", validated.relative_path, validated.sha256)
        )
    finally:
        gateway.close()

    assert failure.status == "pending"
    assert validated.sha256 == created.sha256
    assert prepared.restore_token == "fake-restore-token-1"
    assert prepared.requires_restart is True


def test_process_gateway_rejects_missing_core_backup_capability(tmp_path: Path) -> None:
    supervisor = _test_supervisor(
        _command("--missing-capability", "core_owned_backup"),
        startup_timeout_seconds=2.0,
        core_data_dir=tmp_path,
    )

    with pytest.raises(CoreCapabilityError) as error:
        ProcessCoreGateway(supervisor, required_capabilities=("maintenance",))

    assert error.value.missing_capabilities == ("core_owned_backup",)
    assert supervisor.pid is None


def test_process_stderr_is_captured_without_corrupting_stdout_protocol() -> None:
    supervisor = _test_supervisor(
        _command("--stderr-line", "fake diagnostic"),
        startup_timeout_seconds=2.0,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
        deadline = time.monotonic() + 1.0
        while "fake diagnostic" not in supervisor.stderr_lines():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        stderr_lines = supervisor.stderr_lines()
    finally:
        gateway.close()

    assert health.healthy is True
    assert "fake diagnostic" in stderr_lines


def test_process_gateway_import_does_not_load_python_core() -> None:
    local_root = Path(__file__).resolve().parents[3]
    integrations_root = local_root.parent / "ledgermind-integrations"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(local_root / "src"),
            str(integrations_root / "protocol" / "python" / "src"),
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ledgermind_local.core_gateway.process; "
                "print('ledgermind_core' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "False"


def test_local_runtime_graph_imports_without_python_core() -> None:
    local_root = Path(__file__).resolve().parents[3]
    protocol_root = (
        local_root.parent / "ledgermind-integrations" / "protocol" / "python"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(local_root / "src"), str(protocol_root / "src")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ledgermind_local.bootstrap; "
                "import ledgermind_local.api.app; import ledgermind_local.cli; "
                "import ledgermind_local.projections; import ledgermind_local.search; "
                "print('ledgermind_core' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "False"


def test_process_gateway_handles_object_facet_v2_boundary() -> None:
    supervisor = _test_supervisor(_command(), startup_timeout_seconds=2.0)
    gateway = ProcessCoreGateway(
        supervisor,
        required_capabilities=("object_facet_v2",),
    )

    try:
        ingest = gateway.ingest_raw_round(
            IngestRawRoundCommand(
                command_id="ingest-1",
                idempotency_key="sha256:" + "a" * 64,
                memory_space_id="space-1",
                raw_round_id="raw-1",
                raw_round={},
            )
        )
        tasks = gateway.poll_execution_tasks(
            PollExecutionTasksCommand(
                request_id="poll-1",
                memory_space_id="space-1",
                worker_id="worker-1",
            )
        )
        submitted = gateway.submit_execution_result(
            SubmitExecutionResultCommand(
                request_id="submit-1",
                task_id="task-1",
                memory_space_id="space-1",
                worker_id="worker-1",
                result={"status": "completed"},
            )
        )
        failed = gateway.fail_execution_task(
            FailExecutionTaskCommand(
                request_id="fail-1",
                task_id="task-1",
                memory_space_id="space-1",
                worker_id="worker-1",
                error_code="provider_timeout",
                retryable=True,
                retry_after_seconds=30,
            )
        )
        retrieval = gateway.retrieve_context_v2(
            RetrieveContextV2Command(
                request_id="retrieve-v2-1",
                memory_space_id="space-1",
                query_text="query",
                query_embedding=(0.1,),
            )
        )
        gateway.record_retrieval_outcome_v2(
            RecordRetrievalOutcomeV2Command(
                request_id="outcome-v2-1",
                retrieval_request_id="retrieve-v2-1",
                candidate_value_ids=("value-1",),
                delivered_value_ids=(),
            )
        )
    finally:
        gateway.close()

    assert ingest.accepted is True
    assert ingest.core_raw_round_id == "ingest-1"
    assert tasks.tasks == ()
    assert submitted.accepted is True
    assert failed.released is True
    assert retrieval.payload["retrieval_request_id"] == "retrieve-v2-1"


def test_process_gateway_rejects_mismatched_response_id() -> None:
    supervisor = _test_supervisor(
        _command("--mismatched-health-id"),
        startup_timeout_seconds=2.0,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
        assert supervisor.pid is None
    finally:
        gateway.close()

    assert health.healthy is False
    assert "request_id does not match" in (health.detail or "")


def test_process_gateway_rejects_malformed_response() -> None:
    supervisor = _test_supervisor(
        _command("--malformed-health-response"),
        startup_timeout_seconds=2.0,
    )
    gateway = ProcessCoreGateway(supervisor)

    try:
        health = gateway.health()
        assert supervisor.pid is None
    finally:
        gateway.close()

    assert health.healthy is False
    assert "invalid Core response" in (health.detail or "")
