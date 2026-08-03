from __future__ import annotations

import threading

from ledgermind_local.scheduler.processing_worker import ProcessingWorkerLoop


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
