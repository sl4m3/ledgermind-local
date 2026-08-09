"""Schedulers for local LedgerMind service."""

from .core_command_worker import CoreCommandProcessResult, CoreCommandWorker
from .core_model_task_worker import (
    CoreExecutionTaskWorker,
    CoreModelTaskWorker,
    CoreModelTaskWorkerStats,
)
from .core_projection_worker import CoreProjectionWorker, CoreProjectionWorkerStats
from .guarded_loop import GuardedWorkerLoop
from .retention_worker import RawRoundRetentionWorker, RetentionResult
from .worker_state import WorkerState, WorkerStateSnapshot

__all__ = [
    "CoreCommandProcessResult",
    "CoreCommandWorker",
    "CoreExecutionTaskWorker",
    "CoreModelTaskWorker",
    "CoreModelTaskWorkerStats",
    "CoreProjectionWorker",
    "CoreProjectionWorkerStats",
    "GuardedWorkerLoop",
    "ProcessingWorkerLoop",
    "RawRoundRetentionWorker",
    "RetentionResult",
    "WorkerState",
    "WorkerStateSnapshot",
]


def __getattr__(name: str) -> object:
    """Load the D2-owned legacy processing loop only for legacy callers."""

    if name == "ProcessingWorkerLoop":
        from .processing_worker import ProcessingWorkerLoop

        return ProcessingWorkerLoop
    raise AttributeError(name)
