"""Background loop for durable RawRound processing."""

from __future__ import annotations

from .guarded_loop import GuardedWorkerLoop
from .worker_state import WorkerState


class ProcessingWorkerLoop(GuardedWorkerLoop):
    """Backward-compatible processing loop with exception isolation and state."""

    def __init__(
        self,
        worker: object,
        *,
        poll_interval_seconds: float = 1.0,
        state: WorkerState | None = None,
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            worker,  # type: ignore[arg-type]
            state=state,
            name="processing",
            poll_interval_seconds=poll_interval_seconds,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
