from __future__ import annotations

from types import SimpleNamespace

from ledgermind_local.scheduler import core_execution_task_worker as worker_module
from ledgermind_local.scheduler.core_execution_task_worker import (
    CoreExecutionTaskWorker,
    _execution_result_is_retryable,
)


def test_process_once_closes_discovery_connection_before_provider_work(
    monkeypatch,
) -> None:
    events: list[str] = []

    class Connection:
        closed = False

        def commit(self) -> None:
            events.append("commit")

        def execute(self, _query: str):
            assert not self.closed
            return SimpleNamespace(fetchall=lambda: [("space-1",)])

        def close(self) -> None:
            self.closed = True
            events.append("close")

    connection = Connection()

    class Gateway:
        def require_capabilities(self, _capability: str) -> None:
            return None

        def poll_execution_tasks(self, _command):
            assert connection.closed
            events.append("poll")
            return SimpleNamespace(tasks=[])

    monkeypatch.setattr(
        worker_module.migrations,
        "apply_migrations",
        lambda _connection: events.append("migrate"),
    )
    worker = CoreExecutionTaskWorker(
        database_path="unused.db",
        gateway=Gateway(),
        executor=SimpleNamespace(),
        worker_id="test-worker",
        connection_factory=lambda _path: connection,
    )

    assert worker.process_once() == 0
    assert events == ["migrate", "commit", "close", "poll"]


def test_transient_provider_result_is_retryable() -> None:
    result = SimpleNamespace(status="failed", error_code="transient_provider_error")

    assert _execution_result_is_retryable(result) is True


def test_embedding_provider_unavailable_result_is_retryable() -> None:
    result = SimpleNamespace(status="failed", error_code="provider_unavailable")

    assert _execution_result_is_retryable(result) is True


def test_embedding_transport_result_is_retryable() -> None:
    result = SimpleNamespace(status="failed", error_code="provider_transport_error")

    assert _execution_result_is_retryable(result) is True


def test_schema_failure_result_is_not_retryable() -> None:
    result = SimpleNamespace(status="failed", error_code="schema_shape_failure")

    assert _execution_result_is_retryable(result) is False


def test_schema_failure_gets_one_retry_on_configured_provider_fallback(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    retry_result = SimpleNamespace(status="completed")

    class Gateway:
        def require_capabilities(self, _capability: str) -> None:
            return None

    class Executor:
        def execute(self, _task: object, **kwargs: object) -> object:
            calls.append(kwargs)
            return retry_result

    worker = CoreExecutionTaskWorker(
        database_path="unused.db",
        gateway=Gateway(),
        executor=Executor(),  # type: ignore[arg-type]
        worker_id="test-worker",
    )
    delivered: list[object] = []
    monkeypatch.setattr(
        worker,
        "_deliver_result",
        lambda _task, result, _memory_space_id: delivered.append(result),
    )

    worker._deliver_result_with_structured_retry(
        SimpleNamespace(task_kind="generate_json", task_id="task-1"),
        SimpleNamespace(status="failed", error_code="schema_shape_failure"),
        "space",
    )

    assert calls == [{"force_provider_fallback": True}]
    assert delivered == [retry_result]
