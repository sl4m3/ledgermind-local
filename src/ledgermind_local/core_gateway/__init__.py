"""CoreGateway contracts and the Rust process backend."""

from .base import CoreGateway
from .contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    ContextViewItem,
    ContextViewResult,
    CoreGatewayError,
    CoreHealth,
    DomainRejectedError,
    HypothesisEvidence,
    HypothesisExtraction,
    HypothesisPayload,
    RecordContextUsageCommand,
    RetrieveContextCommand,
    TransientCoreError,
)
from .model_task_contracts import (
    CoreModelTask,
    PollModelTasksCommand,
    PollModelTasksResult,
    SubmitModelResult,
    SubmitModelResultCommand,
)
from .process import ProcessCoreGateway
from .projection_contracts import (
    AckProjectionEventsCommand,
    AckProjectionEventsResult,
    CoreProjectionEvent,
    PollProjectionEventsCommand,
    PollProjectionEventsResult,
)

__all__ = [
    "AcceptHypothesisCommand",
    "AcceptHypothesisResult",
    "AckProjectionEventsCommand",
    "AckProjectionEventsResult",
    "ContextViewItem",
    "ContextViewResult",
    "CoreGateway",
    "CoreGatewayError",
    "CoreHealth",
    "CoreModelTask",
    "CoreProjectionEvent",
    "DomainRejectedError",
    "HypothesisEvidence",
    "HypothesisExtraction",
    "HypothesisPayload",
    "PollModelTasksCommand",
    "PollModelTasksResult",
    "PollProjectionEventsCommand",
    "PollProjectionEventsResult",
    "ProcessCoreGateway",
    "RecordContextUsageCommand",
    "RetrieveContextCommand",
    "SubmitModelResult",
    "SubmitModelResultCommand",
    "TransientCoreError",
]
