"""Local worker for leased Core generative merge tasks."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.model_task_contracts import (
    PollModelTasksCommand,
    SubmitModelResultCommand,
)
from ledgermind_local.inference.schemas import MergeProposal
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations


class MergeProposalBroker(Protocol):
    def generate_merge_proposal(
        self,
        *,
        memory_space_id: str,
        model_input: dict[str, object],
        profile_id: str,
    ) -> MergeProposal: ...


@dataclass(frozen=True, slots=True)
class CoreModelTaskWorkerStats:
    fetched: int = 0
    completed: int = 0
    duplicates: int = 0
    failed: int = 0


ConnectionFactory = Callable[[str | Path], sqlite3.Connection]


class CoreModelTaskWorker:
    """Poll Core leases and execute merge proposals through Local InferenceBroker."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        gateway: CoreGateway,
        broker: MergeProposalBroker,
        worker_id: str,
        poll_limit: int = 10,
        lease_seconds: int = 300,
        connection_factory: ConnectionFactory = open_sqlite_connection,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not 1 <= poll_limit <= 100:
            raise ValueError("poll_limit must be between 1 and 100")
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        self._database_path = str(database_path)
        self._gateway = gateway
        self._broker = broker
        self._worker_id = worker_id
        self._poll_limit = poll_limit
        self._lease_seconds = lease_seconds
        self._connection_factory = connection_factory
        self._closed = False

    def process_once(self) -> CoreModelTaskWorkerStats:
        if self._closed:
            raise RuntimeError("Core model task worker is closed")
        stats = CoreModelTaskWorkerStats()
        connection = self._connection_factory(self._database_path)
        try:
            migrations.apply_migrations(connection)
            rows = connection.execute(
                """
                SELECT memory_space_id
                FROM memory_spaces
                ORDER BY memory_space_id ASC
                """
            ).fetchall()
            for row in rows:
                memory_space_id = str(row[0])
                binding = connection.execute(
                    """
                    SELECT merge_profile_id
                    FROM memory_space_inference_profiles
                    WHERE memory_space_id = ?
                    """,
                    (memory_space_id,),
                ).fetchone()
                merge_profile_id = binding[0] if binding is not None else None
                try:
                    polled = self._gateway.poll_model_tasks(
                        PollModelTasksCommand(
                            request_id=self._request_id("poll"),
                            memory_space_id=memory_space_id,
                            worker_id=self._worker_id,
                            limit=self._poll_limit,
                            lease_seconds=self._lease_seconds,
                        )
                    )
                except Exception:  # noqa: BLE001 - worker boundary must keep leases retryable
                    stats = _add_stats(stats, failed=1)
                    continue
                stats = _add_stats(stats, fetched=len(polled.tasks))
                for task in polled.tasks:
                    if merge_profile_id is None:
                        stats = _add_stats(stats, failed=1)
                        continue
                    try:
                        proposal = self._broker.generate_merge_proposal(
                            memory_space_id=task.memory_space_id,
                            model_input=task.model_input,
                            profile_id=merge_profile_id,
                        )
                        submitted = self._gateway.submit_model_result(
                            SubmitModelResultCommand(
                                request_id=self._request_id("submit"),
                                task_id=task.task_id,
                                memory_space_id=task.memory_space_id,
                                worker_id=self._worker_id,
                                result=proposal.model_dump(mode="json"),
                            )
                        )
                    except Exception:  # noqa: BLE001 - provider transports are heterogeneous
                        stats = _add_stats(stats, failed=1)
                        continue
                    stats = _add_stats(
                        stats,
                        completed=1,
                        duplicates=1 if submitted.duplicate else 0,
                    )
        finally:
            connection.close()
        return stats

    def close(self) -> None:
        self._closed = True

    def _request_id(self, operation: str) -> str:
        return f"{self._worker_id}:{operation}:{uuid.uuid4()}"


def _add_stats(
    stats: CoreModelTaskWorkerStats,
    *,
    fetched: int = 0,
    completed: int = 0,
    duplicates: int = 0,
    failed: int = 0,
) -> CoreModelTaskWorkerStats:
    return CoreModelTaskWorkerStats(
        fetched=stats.fetched + fetched,
        completed=stats.completed + completed,
        duplicates=stats.duplicates + duplicates,
        failed=stats.failed + failed,
    )


__all__ = ["CoreModelTaskWorker", "CoreModelTaskWorkerStats"]
