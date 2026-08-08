"""Persistence package for Local-owned SQLite storage."""

from .database import managed_connection, open_sqlite_connection
from .memory_space_repository import (
    MemorySpace,
    MemorySpaceSourceClientChangedError,
    SQLiteMemorySpaceRepository,
)
from .raw_round_repository import (
    CoreCommandRecord,
    CoreRawRoundDeliveryRecord,
    NormalizedRoundRecord,
    RawRoundRecord,
    RoundProcessingJob,
    SQLiteRawRoundRepository,
)
from .sqlite_uow import (
    SQLiteUnitOfWork,
    SQLiteUnitOfWorkError,
    SQLiteUnitOfWorkInactiveError,
)

__all__ = [
    "CoreCommandRecord",
    "CoreRawRoundDeliveryRecord",
    "MemorySpace",
    "MemorySpaceSourceClientChangedError",
    "NormalizedRoundRecord",
    "RawRoundRecord",
    "RoundProcessingJob",
    "SQLiteMemorySpaceRepository",
    "SQLiteRawRoundRepository",
    "SQLiteUnitOfWork",
    "SQLiteUnitOfWorkError",
    "SQLiteUnitOfWorkInactiveError",
    "managed_connection",
    "open_sqlite_connection",
]
