from __future__ import annotations

import threading

from ledgermind_local.scheduler.guarded_loop import GuardedWorkerLoop
from ledgermind_local.scheduler.processing_worker import ProcessingWorkerLoop
from ledgermind_local.scheduler.worker_state import WorkerState


def test_processing_worker_loop_stops_without_busy_spin() -> None:
    calls = {"count": 0}
    entered = threading.Event()

    class Worker:
        def process_once(self):
            calls["count"] += 1
            entered.set()

    loop = ProcessingWorkerLoop(Worker(), poll_interval_seconds=0.01)
    thread = threading.Thread(target=loop.run)
    thread.start()
    assert entered.wait(timeout=1)
    loop.request_stop()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert calls["count"] >= 1


def test_guarded_loop_survives_error_and_records_recovery() -> None:
    state = WorkerState("test-worker")
    failed = threading.Event()
    succeeded = threading.Event()

    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def process_once(self):
            self.calls += 1
            if self.calls == 1:
                failed.set()
                raise RuntimeError("payload must not be logged")
            succeeded.set()
            loop.request_stop()
            return type("Result", (), {"job_id": "job-1"})()

    worker = Worker()
    loop = GuardedWorkerLoop(
        worker,
        state=state,
        poll_interval_seconds=0,
        initial_backoff_seconds=0,
        max_backoff_seconds=0,
    )
    loop.start()

    assert failed.wait(timeout=1)
    assert succeeded.wait(timeout=1)
    assert loop.join(timeout=1) is True
    snapshot = state.snapshot()
    assert snapshot.running is False
    assert snapshot.healthy is True
    assert snapshot.last_error_code == "RuntimeError"
    assert snapshot.consecutive_failures == 0
    assert snapshot.processed_count == 1
    assert snapshot.failed_count == 1
    assert snapshot.current_item_id is None


def test_guarded_loop_stop_does_not_start_another_iteration() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    class Worker:
        def process_once(self):
            calls["count"] += 1
            entered.set()
            assert release.wait(timeout=1)

    loop = GuardedWorkerLoop(
        Worker(),
        poll_interval_seconds=0,
        initial_backoff_seconds=0,
        max_backoff_seconds=0,
    )
    loop.start()
    assert entered.wait(timeout=1)
    loop.request_stop()
    release.set()

    assert loop.join(timeout=1) is True
    assert calls["count"] == 1
