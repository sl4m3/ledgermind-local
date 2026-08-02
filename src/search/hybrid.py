"""Hybrid search adapter combining FTS and optional vector retrieval."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ports import KnowledgeSearch, SearchHit

from projections import GGUFVectorizer, VectorProjectionStore
from projections.vectorizer import Vectorizer
from search.fts import SQLiteKnowledgeSearchAdapter

__all__ = [
    "HybridKnowledgeSearchAdapter",
]


def _normalize_vector_score(similarity: float) -> float:
    if similarity <= -1.0:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return 0.5 * (similarity + 1.0)


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("embedding must be a sequence of numbers")
    return [float(item) for item in value]


def _vector_norm(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value) ** 2
    return math.sqrt(total)


def _vector_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")

    if not left or not right:
        return 0.0

    numerator = 0.0
    for a, b in zip(left, right):
        numerator += float(a) * float(b)

    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return numerator / (left_norm * right_norm)


def _current_knowledge_ids(
    connection: sqlite3.Connection,
    memory_space_id: str,
    knowledge_ids: Sequence[str],
) -> set[str]:
    ids = tuple(knowledge_id for knowledge_id in knowledge_ids if knowledge_id)
    if not ids:
        return set()
    placeholders = ", ".join(["?"] * len(ids))
    rows = connection.execute(
        f"""
        SELECT knowledge_id
        FROM knowledge_items
        WHERE memory_space_id = ?
          AND knowledge_id IN ({placeholders})
          AND superseded_by_id IS NULL
          AND deleted_at IS NULL
        """,
        (memory_space_id, *ids),
    ).fetchall()
    return {row["knowledge_id"] for row in rows}


class _NoopVectorizerError(Exception):
    pass


class HybridKnowledgeSearchAdapter(KnowledgeSearch):
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        vector_store_root: str | Path | None = None,
        vectorizer_factory: Callable[[], Vectorizer] | None = None,
    ) -> None:
        self._connection = connection
        self._vector_store_root = Path(vector_store_root) if vector_store_root is not None else None
        self._vectorizer_factory = vectorizer_factory
        self._fts = SQLiteKnowledgeSearchAdapter(connection)
        self._vector_store: VectorProjectionStore | None = None
        self._vectorizer: Vectorizer | None = None
        self._vector_disabled = False

    @property
    def vector_store(self) -> VectorProjectionStore | None:
        root = self._vector_store_root
        if self._vector_store is not None or root is None:
            return self._vector_store
        if not root.exists():
            return None
        self._vector_store = VectorProjectionStore(root)
        return self._vector_store

    @property
    def _vectorizer_instance(self) -> Vectorizer | None:
        if self._vector_disabled:
            return None

        if self._vectorizer is not None:
            return self._vectorizer

        if self._vectorizer_factory is None:
            store = self.vector_store
            if store is None:
                self._vector_disabled = True
                return None
            manifest = dict(store.manifest)
            if manifest.get("model_name") is None:
                self._vector_disabled = True
                return None
            self._vectorizer_factory = lambda: GGUFVectorizer(manifest=manifest)

        try:
            self._vectorizer = self._vectorizer_factory()
        except Exception as exc:  # pragma: no cover - defensive
            self._vector_disabled = True
            raise _NoopVectorizerError(str(exc)) from exc
        return self._vectorizer

    def _build_vector_hits(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
    ) -> list[SearchHit]:
        if limit <= 0 or not query:
            return []

        store = self.vector_store
        if store is None or store.document_count == 0:
            return []

        try:
            vectorizer = self._vectorizer_instance
        except _NoopVectorizerError:
            return []
        if vectorizer is None:
            return []

        try:
            vectors = vectorizer.encode([query])
        except Exception:
            self._vector_disabled = True
            return []

        vectors_list = list(vectors)
        if len(vectors_list) != 1:
            return []

        query_vector = _coerce_vector(vectors_list[0])
        if not query_vector:
            return []

        knowledge_ids = store.ids
        raw_vectors = store.vectors
        if len(knowledge_ids) != len(raw_vectors):
            return []

        current_ids = _current_knowledge_ids(self._connection, memory_space_id, knowledge_ids)
        if not current_ids:
            return []

        hits: list[tuple[float, str]] = []
        for knowledge_id, raw_vector in zip(knowledge_ids, raw_vectors):
            if knowledge_id not in current_ids:
                continue
            try:
                vector = _coerce_vector(raw_vector)
                score = _normalize_vector_score(_vector_similarity(query_vector, vector))
            except Exception:
                continue
            hits.append((score, knowledge_id))

        hits.sort(key=lambda item: (-item[0], item[1]))
        ranked = []
        for score, knowledge_id in hits[:limit]:
            ranked.append(
                SearchHit(
                    knowledge_id=knowledge_id,
                    lexical_score=0.0,
                    vector_score=score,
                )
            )
        return ranked

    def search(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SearchHit]:
        if not query or limit <= 0 or offset < 0:
            return []

        required = offset + limit
        if required <= 0:
            return []

        lexical_hits = self._fts.search(
            memory_space_id=memory_space_id,
            query=query,
            limit=required,
        )
        vector_hits = self._build_vector_hits(
            memory_space_id=memory_space_id,
            query=query,
            limit=required,
        )

        merged: dict[str, SearchHit] = {}
        for hit in lexical_hits:
            merged[hit.knowledge_id] = hit
        for hit in vector_hits:
            existing = merged.get(hit.knowledge_id)
            if existing is None:
                merged[hit.knowledge_id] = hit
            else:
                merged[hit.knowledge_id] = SearchHit(
                    knowledge_id=existing.knowledge_id,
                    lexical_score=existing.lexical_score,
                    vector_score=hit.vector_score,
                )

        ranked = list(merged.values())
        return ranked[offset : offset + limit]
