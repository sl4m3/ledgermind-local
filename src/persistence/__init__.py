"""Persistence package for local LedgerMind SQLite storage."""

from .atom_repository import Atom, SQLiteAtomRepository
from .database import managed_connection, open_sqlite_connection
from .evidence_repository import KnowledgeEvidence, SQLiteEvidenceRepository
from .idempotency_repository import SQLiteIdempotencyRepository, StoredIdempotencyResult
from .knowledge_repository import (
    Knowledge,
    SQLiteKnowledgeConcurrencyError,
    SQLiteKnowledgeRepository,
)
from .memory_space_repository import (
    MemorySpace,
    MemorySpaceSourceClientChangedError,
    SQLiteMemorySpaceRepository,
)
from .outbox_repository import OutboxEvent, SQLiteOutboxRepository
from .revision_repository import KnowledgeRevision, SQLiteRevisionRepository
from .sqlite_uow import (
    SQLiteUnitOfWork,
    SQLiteUnitOfWorkError,
    SQLiteUnitOfWorkInactiveError,
)

__all__ = [
    "Atom",
    "Knowledge",
    "KnowledgeEvidence",
    "KnowledgeRevision",
    "MemorySpace",
    "MemorySpaceSourceClientChangedError",
    "OutboxEvent",
    "SQLiteAtomRepository",
    "SQLiteEvidenceRepository",
    "SQLiteIdempotencyRepository",
    "SQLiteKnowledgeConcurrencyError",
    "SQLiteKnowledgeRepository",
    "SQLiteMemorySpaceRepository",
    "SQLiteOutboxRepository",
    "SQLiteRevisionRepository",
    "SQLiteUnitOfWork",
    "SQLiteUnitOfWorkError",
    "SQLiteUnitOfWorkInactiveError",
    "StoredIdempotencyResult",
    "managed_connection",
    "open_sqlite_connection",
]
