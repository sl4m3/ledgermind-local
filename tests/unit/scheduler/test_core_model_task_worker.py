from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.core_gateway.model_task_contracts import (
    CoreModelTask,
    FailModelTaskResult,
    PollModelTasksResult,
    SubmitModelResult,
)
from ledgermind_local.inference.schemas import MergeProposal
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.scheduler.core_model_task_worker import CoreModelTaskWorker


class _Gateway:
    def __init__(
        self,
        task: CoreModelTask | tuple[CoreModelTask, ...],
        *,
        failure_status: str | None = None,
        poll_error: Exception | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        self.tasks = task if isinstance(task, tuple) else (task,)
        self.failure_status = failure_status
        self.poll_error = poll_error
        self.submit_error = submit_error
        self.polls = 0
        self.submissions = []
        self.failures = []

    def poll_model_tasks(self, command):
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        if self.polls > 1:
            return PollModelTasksResult(tasks=(), has_more=False)
        return PollModelTasksResult(tasks=self.tasks, has_more=False)

    def submit_model_result(self, command):
        if self.submit_error is not None:
            raise self.submit_error
        self.submissions.append(command)
        return SubmitModelResult(accepted=True, duplicate=True, status="completed")

    def fail_model_task(self, command):
        self.failures.append(command)
        status = self.failure_status or ("pending" if command.retryable else "failed")
        return FailModelTaskResult(
            status=status,
            attempts=1,
            available_at=command.failed_at if status == "pending" else None,
            last_error_code=command.error_code,
            failed_at=command.failed_at,
            completed_at=None if status == "pending" else command.failed_at,
        )


class _Broker:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        errors: list[Exception | None] | None = None,
    ) -> None:
        self.error = error
        self.errors = list(errors or [])
        self.calls = []

    def generate_merge_proposal(self, **kwargs):
        self.calls.append(kwargs)
        error = self.errors.pop(0) if self.errors else self.error
        if error is not None:
            raise error
        return MergeProposal(
            title="Merged",
            target="ops",
            statement="Combined statement",
            rationale="Agreement",
            preserved_references=("knowledge-a", "knowledge-b"),
            preserved_constraints=("keep source",),
        )


def _database(tmp_path):
    database = tmp_path / "rounds.db"
    connection = sqlite3.connect(database)
    migrations.apply_migrations(connection)
    connection.execute(
        "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("space-1", "tests", "2026-08-03T00:00:00Z", "2026-08-03T00:00:00Z"),
    )
    connection.execute(
        """
        INSERT INTO memory_space_inference_profiles(
            memory_space_id, hypothesis_profile_id, merge_profile_id, updated_at
        ) VALUES (?, ?, ?, ?)
        """,
        ("space-1", None, "merge-default", "2026-08-03T00:00:00Z"),
    )
    connection.commit()
    connection.close()
    return database


def _task() -> CoreModelTask:
    return CoreModelTask(
        task_id="task-1",
        operation="merge_knowledge",
        memory_space_id="space-1",
        expected_versions={"knowledge-a": 1, "knowledge-b": 2},
        expires_at="2026-08-03T12:00:00Z",
        model_input={"items": [{"reference": "knowledge-a"}]},
        lease_expires_at="2026-08-03T11:05:00Z",
    )


def test_model_task_worker_executes_and_submits_merge_proposal(tmp_path) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(_task())
    broker = _Broker()
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=broker,
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.fetched == 1
    assert stats.completed == 1
    assert stats.duplicates == 1
    assert stats.failed == 0
    assert broker.calls[0]["profile_id"] == "merge-default"
    assert gateway.submissions[0].result["title"] == "Merged"


def test_model_task_worker_does_not_submit_provider_failure(tmp_path) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(_task())
    broker = _Broker(error=RuntimeError("provider unavailable"))
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=broker,
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.fetched == 1
    assert stats.completed == 0
    assert stats.failed == 1
    assert gateway.submissions == []
    assert len(gateway.failures) == 1
    assert gateway.failures[0].error_code == "provider_unavailable"
    assert gateway.failures[0].retryable is True
    assert stats.retry_scheduled == 1
    assert stats.terminal_failures == 0
    assert stats.provider_failures == 1
    assert stats.last_error_code == "provider_unavailable"
    assert stats.degraded is True
    assert stats.made_progress is True


def test_model_task_worker_uses_core_terminal_status_when_retry_limit_is_exhausted(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(_task(), failure_status="failed")
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=_Broker(error=RuntimeError("provider temporarily unavailable")),
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert gateway.failures[0].retryable is True
    assert stats.retry_scheduled == 0
    assert stats.terminal_failures == 1
    assert stats.permanent_failures == 0
    assert stats.made_progress is True


def test_model_task_worker_classifies_poll_failure_as_core_failure(tmp_path) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(
        _task(),
        poll_error=RuntimeError("provider unavailable; token=secret"),
    )
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=_Broker(),
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.failed == 1
    assert stats.core_poll_failures == 1
    assert stats.provider_failures == 0
    assert stats.core_delivery_failures == 0
    assert stats.last_error_code == "core_poll_error"
    assert stats.made_progress is False
    assert "secret" not in repr(stats)


def test_model_task_worker_reports_submit_failure_back_to_core(tmp_path) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(
        _task(),
        submit_error=RuntimeError("core transport token=secret"),
    )
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=_Broker(),
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert len(gateway.failures) == 1
    assert gateway.failures[0].error_code == "core_delivery_error"
    assert gateway.failures[0].retryable is True
    assert stats.core_delivery_failures == 1
    assert stats.provider_failures == 0
    assert stats.retry_scheduled == 1
    assert stats.last_error_code == "core_delivery_error"
    assert "secret" not in repr(stats)
    assert "secret" not in repr(gateway.failures[0].to_payload())


def test_model_task_worker_partial_success_remains_degraded(tmp_path) -> None:
    database = _database(tmp_path)
    second_task = CoreModelTask(
        task_id="task-2",
        operation="merge_knowledge",
        memory_space_id="space-1",
        expected_versions={"knowledge-a": 1, "knowledge-b": 2},
        expires_at="2026-08-03T12:00:00Z",
        model_input={"items": [{"reference": "knowledge-b"}]},
        lease_expires_at="2026-08-03T11:05:00Z",
    )
    gateway = _Gateway((_task(), second_task))
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=_Broker(errors=[None, RuntimeError("provider unavailable")]),
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.completed == 1
    assert stats.failed == 1
    assert stats.degraded is True
    assert stats.made_progress is True
    assert stats.last_error_code == "provider_unavailable"


def test_model_task_worker_releases_invalid_provider_response_as_permanent(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    gateway = _Gateway(_task())
    broker = _Broker(error=ValueError("response schema is invalid"))
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=broker,
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.failed == 1
    assert len(gateway.failures) == 1
    assert gateway.failures[0].error_code == "invalid_model_output"
    assert gateway.failures[0].retryable is False
    assert stats.permanent_failures == 1
    assert stats.terminal_failures == 1
    assert stats.retry_scheduled == 0
    assert stats.provider_failures == 1


def test_model_task_worker_releases_task_when_profile_is_missing(tmp_path) -> None:
    database = _database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "DELETE FROM memory_space_inference_profiles WHERE memory_space_id = ?",
        ("space-1",),
    )
    connection.commit()
    connection.close()
    gateway = _Gateway(_task())
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=gateway,
        broker=_Broker(),
        worker_id="local-model-tasks",
    )

    stats = worker.process_once()

    assert stats.failed == 1
    assert len(gateway.failures) == 1
    assert gateway.failures[0].error_code == "profile_missing"
    assert gateway.failures[0].retryable is False


def test_model_task_worker_rejects_use_after_close(tmp_path) -> None:
    database = _database(tmp_path)
    worker = CoreModelTaskWorker(
        database_path=database,
        gateway=_Gateway(_task()),
        broker=_Broker(),
        worker_id="local-model-tasks",
    )
    worker.close()

    with pytest.raises(RuntimeError, match="closed"):
        worker.process_once()
