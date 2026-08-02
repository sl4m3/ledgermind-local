"""Persistence package for local LedgerMind SQLite storage."""

from .database import managed_connection, open_sqlite_connection
from .atom_repository import Atom, SQLiteAtomRepository
from .evidence_repository import KnowledgeEvidence, SQLiteEvidenceRepository
from .idempotency_repository import StoredIdempotencyResult, SQLiteIdempotencyRepository
from .knowledge_repository import Knowledge, SQLiteKnowledgeRepository, SQLiteKnowledgeConcurrencyError
from .outbox_repository import OutboxEvent, SQLiteOutboxRepository
from .revision_repository import KnowledgeRevision, SQLiteRevisionRepository
from .memory_space_repository import (
    MemorySpace,
    MemorySpaceSourceClientChangedError,
    SQLiteMemorySpaceRepository,
)
from .sqlite_uow import (
    SQLiteUnitOfWork,
    SQLiteUnitOfWorkError,
    SQLiteUnitOfWorkInactiveError,
)

__all__ = [
    "managed_connection",
    "open_sqlite_connection",
    "Atom",
    "SQLiteAtomRepository",
    "KnowledgeEvidence",
    "SQLiteEvidenceRepository",
    "StoredIdempotencyResult",
    "SQLiteIdempotencyRepository",
    "Knowledge",
    "SQLiteKnowledgeConcurrencyError",
    "SQLiteKnowledgeRepository",
    "OutboxEvent",
    "SQLiteOutboxRepository",
    "KnowledgeRevision",
    "SQLiteRevisionRepository",
    "MemorySpace",
    "MemorySpaceSourceClientChangedError",
    "SQLiteMemorySpaceRepository",
    "SQLiteUnitOfWork",
    "SQLiteUnitOfWorkError",
    "SQLiteUnitOfWorkInactiveError",
]
