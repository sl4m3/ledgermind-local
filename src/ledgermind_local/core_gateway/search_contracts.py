"""Local-owned search port contracts for projection adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One public search ranking result produced by a Local projection."""

    knowledge_id: str
    lexical_score: float
    vector_score: float | None


class KnowledgeSearch(Protocol):
    """Port implemented by Local-owned search projection adapters."""

    def search(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
    ) -> list[SearchHit]:
        """Return ranked current knowledge projection hits."""


__all__ = ["KnowledgeSearch", "SearchHit"]
