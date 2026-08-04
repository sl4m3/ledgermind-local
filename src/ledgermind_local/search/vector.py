"""Memory-space-scoped vector candidate search for the Core projection."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from typing_extensions import Self

from ledgermind_local.projections.vector_store import VectorProjectionStore
from ledgermind_local.projections.vectorizer import Vectorizer

from .core_backed import CandidateScore, LocalCandidateSearch

_FTS_TABLE = "core_knowledge_fts"


class CoreProjectionVectorSearchAdapter(LocalCandidateSearch):
    """Search the Local vector index while scoping IDs through the FTS projection.

    The vector store intentionally persists only knowledge IDs.  The event-derived
    FTS projection is therefore used as the memory-space membership index; a vector
    ID without a row in the requested space is never returned to Core.
    """

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        vector_store_root: str | Path,
        vectorizer_factory: Callable[[], Vectorizer],
    ) -> None:
        self._connection = connection
        self._vector_store = VectorProjectionStore(Path(vector_store_root))
        self._vectorizer_factory = vectorizer_factory
        self._vectorizer_instance: Vectorizer | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._vectorizer_instance is not None:
            self._vectorizer_instance.close()
            self._vectorizer_instance = None

    def search(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
    ) -> Sequence[CandidateScore]:
        if limit <= 0 or not query.strip():
            return ()
        if not self._table_exists():
            raise RuntimeError("Core FTS projection is unavailable for vector scoping")

        allowed_ids = {
            str(row[0])
            for row in self._connection.execute(
                f"SELECT knowledge_id FROM {_FTS_TABLE} WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchall()
        }
        if not allowed_ids:
            return ()

        query_vector = self._encode_query(query)
        candidates: list[CandidateScore] = []
        for knowledge_id, vector in zip(
            self._vector_store.ids,
            self._vector_store.vectors,
            strict=True,
        ):
            if knowledge_id not in allowed_ids:
                continue
            candidates.append(
                CandidateScore(
                    knowledge_id=knowledge_id,
                    score=_cosine_score(query_vector, vector),
                    source="vector",
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.knowledge_id))
        return tuple(candidates[:limit])

    @property
    def _vectorizer(self) -> Vectorizer:
        if self._vectorizer_instance is None:
            self._vectorizer_instance = self._vectorizer_factory()
        return self._vectorizer_instance

    def _encode_query(self, query: str) -> tuple[float, ...]:
        vectors = list(self._vectorizer.encode([query]))
        if len(vectors) != 1:
            raise ValueError("partial vectorization result")
        return _coerce_vector(vectors[0])

    def _table_exists(self) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ? LIMIT 1
            """,
            (_FTS_TABLE,),
        ).fetchone()
        return row is not None


def _coerce_vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("embedding must be a sequence of numbers")
    vector = tuple(float(item) for item in value)
    if not vector:
        raise ValueError("embedding must not be empty")
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("embedding must contain finite values")
    return vector


def _cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    right_vector = _coerce_vector(right)
    if len(left) != len(right_vector):
        raise ValueError("query and projection vector dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right_vector, strict=True)) / (
        left_norm * right_norm
    )
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


VectorProjectionSearchAdapter = CoreProjectionVectorSearchAdapter

__all__ = [
    "CoreProjectionVectorSearchAdapter",
    "VectorProjectionSearchAdapter",
]
