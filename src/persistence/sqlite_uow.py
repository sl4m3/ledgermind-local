"""SQLite unit-of-work helper."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import ClassVar

from typing_extensions import Self

from .atom_repository import SQLiteAtomRepository
from .database import open_sqlite_connection
from .evidence_repository import SQLiteEvidenceRepository
from .idempotency_repository import SQLiteIdempotencyRepository
from .knowledge_repository import SQLiteKnowledgeRepository
from .memory_space_repository import SQLiteMemorySpaceRepository
from .outbox_repository import SQLiteOutboxRepository
from .revision_repository import SQLiteRevisionRepository


class SQLiteUnitOfWorkError(RuntimeError):
    """Base error for unit-of-work misuse."""


class SQLiteUnitOfWorkInactiveError(SQLiteUnitOfWorkError):
    """Raised when unit-of-work is used outside an active transaction scope."""


@dataclass
class SQLiteUnitOfWork:
    """Transaction boundary that owns exactly one connection and one transaction."""

    database_path: str | Path
    busy_timeout_ms: int = 5_000

    _connection: sqlite3.Connection | None = None
    _committed: bool = False
    _atoms: SQLiteAtomRepository | None = None
    _knowledge: SQLiteKnowledgeRepository | None = None
    _evidence: SQLiteEvidenceRepository | None = None
    _revisions: SQLiteRevisionRepository | None = None
    _idempotency: SQLiteIdempotencyRepository | None = None
    _outbox: SQLiteOutboxRepository | None = None
    _memory_spaces: SQLiteMemorySpaceRepository | None = None

    _closed_error_message: ClassVar[str] = "sqlite unit of work is not active"

    def __enter__(self) -> Self:
        if self._connection is not None:
            return self

        self._connection = open_sqlite_connection(
            self.database_path,
            busy_timeout_ms=self.busy_timeout_ms,
        )
        self._committed = False
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._atoms = SQLiteAtomRepository(self._connection)
            self._knowledge = SQLiteKnowledgeRepository(self._connection)
            self._evidence = SQLiteEvidenceRepository(self._connection)
            self._revisions = SQLiteRevisionRepository(self._connection)
            self._idempotency = SQLiteIdempotencyRepository(self._connection)
            self._outbox = SQLiteOutboxRepository(self._connection)
            self._memory_spaces = SQLiteMemorySpaceRepository(self._connection)
            return self
        except Exception:
            self._disconnect()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                if not self._committed:
                    self.rollback()
            else:
                self.rollback()
        finally:
            self._disconnect()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._connection

    @property
    def atoms(self) -> SQLiteAtomRepository:
        if self._atoms is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._atoms

    @property
    def knowledge(self) -> SQLiteKnowledgeRepository:
        if self._knowledge is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._knowledge

    @property
    def evidence(self) -> SQLiteEvidenceRepository:
        if self._evidence is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._evidence

    @property
    def revisions(self) -> SQLiteRevisionRepository:
        if self._revisions is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._revisions

    @property
    def idempotency(self) -> SQLiteIdempotencyRepository:
        if self._idempotency is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._idempotency

    @property
    def outbox(self) -> SQLiteOutboxRepository:
        if self._outbox is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._outbox

    @property
    def memory_spaces(self) -> SQLiteMemorySpaceRepository:
        if self._memory_spaces is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        return self._memory_spaces

    def commit(self) -> None:
        if self._connection is None:
            raise SQLiteUnitOfWorkInactiveError(self._closed_error_message)
        if self._committed:
            raise SQLiteUnitOfWorkError("commit already called")

        self._connection.commit()
        self._committed = True

    def rollback(self) -> None:
        if self._connection is None:
            return
        if self._committed:
            return
        self._connection.rollback()

    def _disconnect(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None
            self._committed = False
            self._atoms = None
            self._knowledge = None
            self._evidence = None
            self._revisions = None
            self._idempotency = None
            self._outbox = None
            self._memory_spaces = None
