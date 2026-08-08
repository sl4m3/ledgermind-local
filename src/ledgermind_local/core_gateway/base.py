"""Boundary protocol for the future isolated Core backend."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    AcceptHypothesisCommand,
    AcceptHypothesisResult,
    ContextViewResult,
    CoreHealth,
    FailExecutionTaskCommand,
    FailExecutionTaskResult,
    IngestRawRoundCommand,
    IngestRawRoundResult,
    PollExecutionTasksCommand,
    PollExecutionTasksResult,
    RecordContextUsageCommand,
    RecordRetrievalOutcomeV2Command,
    RetrieveContextCommand,
    RetrieveContextV2Command,
    RetrieveContextV2Result,
    SubmitExecutionResult,
    SubmitExecutionResultCommand,
)
from .maintenance import (
    BackupManifest,
    BeginRestoreCommand,
    BeginRestoreResult,
    CommitRestoreCommand,
    CommitRestoreResult,
    CreateBackupCommand,
    PrepareRestoreCommand,
    PrepareRestoreResult,
    RollbackRestoreCommand,
    RollbackRestoreResult,
    ValidateBackupCommand,
)
from .model_task_contracts import (
    FailModelTaskCommand,
    FailModelTaskResult,
    PollModelTasksCommand,
    PollModelTasksResult,
    SubmitModelResult,
    SubmitModelResultCommand,
)
from .projection_contracts import (
    AckProjectionEventsCommand,
    AckProjectionEventsResult,
    PollProjectionEventsCommand,
    PollProjectionEventsResult,
)


class CoreGateway(Protocol):
    """Only supported Local boundary for Core operations."""

    def accept_hypothesis(
        self,
        command: AcceptHypothesisCommand,
    ) -> AcceptHypothesisResult: ...

    def ingest_raw_round(
        self, command: IngestRawRoundCommand
    ) -> IngestRawRoundResult: ...

    def retrieve_context(
        self,
        request: RetrieveContextCommand,
    ) -> ContextViewResult: ...

    def record_context_usage(self, command: RecordContextUsageCommand) -> None: ...

    def retrieve_context_v2(
        self, request: RetrieveContextV2Command
    ) -> RetrieveContextV2Result: ...

    def record_retrieval_outcome_v2(
        self, command: RecordRetrievalOutcomeV2Command
    ) -> None: ...

    def poll_execution_tasks(
        self, command: PollExecutionTasksCommand
    ) -> PollExecutionTasksResult: ...

    def submit_execution_result(
        self, command: SubmitExecutionResultCommand
    ) -> SubmitExecutionResult: ...

    def fail_execution_task(
        self, command: FailExecutionTaskCommand
    ) -> FailExecutionTaskResult: ...

    def poll_projection_events(
        self, command: PollProjectionEventsCommand
    ) -> PollProjectionEventsResult: ...

    def ack_projection_events(
        self, command: AckProjectionEventsCommand
    ) -> AckProjectionEventsResult: ...

    def poll_model_tasks(
        self, command: PollModelTasksCommand
    ) -> PollModelTasksResult: ...

    def submit_model_result(
        self, command: SubmitModelResultCommand
    ) -> SubmitModelResult: ...

    def fail_model_task(
        self, command: FailModelTaskCommand
    ) -> FailModelTaskResult: ...

    def create_backup(self, command: CreateBackupCommand) -> BackupManifest: ...

    def validate_backup(self, command: ValidateBackupCommand) -> BackupManifest: ...

    def prepare_restore(
        self, command: PrepareRestoreCommand
    ) -> PrepareRestoreResult: ...

    def begin_restore(self, command: BeginRestoreCommand) -> BeginRestoreResult: ...

    def commit_restore(self, command: CommitRestoreCommand) -> CommitRestoreResult: ...

    def rollback_restore(self, command: RollbackRestoreCommand) -> RollbackRestoreResult: ...

    def health(self) -> CoreHealth: ...


__all__ = ["CoreGateway"]
