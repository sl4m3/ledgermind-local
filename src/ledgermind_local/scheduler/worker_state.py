"""Thread-safe runtime state for Local background workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from threading import RLock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class WorkerStateSnapshot:
    """Immutable point-in-time view of a worker's runtime state."""

    name: str
    running: bool = False
    healthy: bool = True
    last_started_at: str | None = None
    last_success_at: str | None = None
    last_error_at: str | None = None
    last_error_code: str | None = None
    consecutive_failures: int = 0
    processed_count: int = 0
    failed_count: int = 0
    current_item_id: str | None = None


class WorkerState:
    """Mutable worker state whose snapshots are safe across threads."""

    def __init__(self, name: str) -> None:
        if not name.strip():
            raise ValueError("worker state name must not be empty")
        self._lock = RLock()
        self._snapshot = WorkerStateSnapshot(name=name)

    def snapshot(self) -> WorkerStateSnapshot:
        """Return an immutable copy of the current state."""

        with self._lock:
            return replace(self._snapshot)

    @property
    def name(self) -> str:
        return self.snapshot().name

    @property
    def running(self) -> bool:
        return self.snapshot().running

    @property
    def healthy(self) -> bool:
        return self.snapshot().healthy

    @property
    def last_started_at(self) -> str | None:
        return self.snapshot().last_started_at

    @property
    def last_success_at(self) -> str | None:
        return self.snapshot().last_success_at

    @property
    def last_error_at(self) -> str | None:
        return self.snapshot().last_error_at

    @property
    def last_error_code(self) -> str | None:
        return self.snapshot().last_error_code

    @property
    def consecutive_failures(self) -> int:
        return self.snapshot().consecutive_failures

    @property
    def processed_count(self) -> int:
        return self.snapshot().processed_count

    @property
    def failed_count(self) -> int:
        return self.snapshot().failed_count

    @property
    def current_item_id(self) -> str | None:
        return self.snapshot().current_item_id

    def mark_started(self, *, item_id: str | None = None) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=True,
                last_started_at=_now(),
                current_item_id=item_id,
            )

    def set_current_item(self, item_id: str | None) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, current_item_id=item_id)

    def mark_success(
        self,
        *,
        item_id: str | None = None,
        processed: bool = True,
    ) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=True,
                healthy=True,
                last_success_at=_now(),
                consecutive_failures=0,
                processed_count=(
                    self._snapshot.processed_count + (1 if processed else 0)
                ),
                current_item_id=item_id,
            )

    def mark_error(self, error_code: str, *, item_id: str | None = None) -> None:
        if not error_code.strip():
            raise ValueError("worker error code must not be empty")
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=True,
                healthy=False,
                last_error_at=_now(),
                last_error_code=error_code,
                consecutive_failures=self._snapshot.consecutive_failures + 1,
                failed_count=self._snapshot.failed_count + 1,
                current_item_id=item_id,
            )

    def mark_stopped(self) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                running=False,
                current_item_id=None,
            )

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly snapshot without mutable internals."""

        return asdict(self.snapshot())


__all__ = ["WorkerState", "WorkerStateSnapshot"]
