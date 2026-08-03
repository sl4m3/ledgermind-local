"""DI containers and request-scoped defaults for the local API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Application(Protocol):
    """Dependency interface expected by API routes."""

    def build_ingest_atom_handler(self) -> object:
        """Return an ``IngestAtomHandler``-like callable object."""

    def build_get_atom_handler(self) -> object:
        """Return a ``GetAtomHandler``-like callable object."""

    def build_get_knowledge_handler(self) -> object:
        """Return a ``GetKnowledgeHandler``-like callable object."""

    def build_get_knowledge_history_handler(self) -> object:
        """Return a ``get knowledge history`` handler."""

    def build_get_knowledge_evidence_handler(self) -> object:
        """Return a ``get knowledge evidence`` handler."""

    def build_retrieve_context_handler(self) -> object:
        """Return a ``RetrieveContextHandler``-like callable object."""

    def build_ingest_raw_round_handler(self) -> object:
        """Return an immutable RawRound capture handler."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for API operation."""

    database_path: str | Path
    api_token: str | None = None
    service_lock_path: Path | None = None
    max_raw_round_bytes: int = 5_000_000
    raw_round_retention_days: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", Path(self.database_path))
        if self.max_raw_round_bytes < 1:
            raise ValueError("max_raw_round_bytes must be positive")
        if self.raw_round_retention_days < 1:
            raise ValueError("raw_round_retention_days must be positive")
        lock_path = self.service_lock_path
        object.__setattr__(
            self,
            "service_lock_path",
            Path(lock_path) if lock_path is not None else None,
        )
