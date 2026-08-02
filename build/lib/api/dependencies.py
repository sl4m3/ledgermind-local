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

    def build_retrieve_context_handler(self) -> object:
        """Return a ``RetrieveContextHandler``-like callable object."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for API operation."""

    database_path: str | Path
    api_token: str | None = None
    service_lock_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", Path(self.database_path))
        lock_path = self.service_lock_path
        object.__setattr__(
            self,
            "service_lock_path",
            Path(lock_path) if lock_path is not None else None,
        )
