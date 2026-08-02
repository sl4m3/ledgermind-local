"""Full-text search adapters built on top of the SQLite FTS projection."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from ports import KnowledgeSearch, SearchHit

from projections.fts import KnowledgeFTSProjection

__all__ = [
    "RawSQLiteKnowledgeSearchAdapter",
    "SQLiteKnowledgeSearchAdapter",
]


def _normalize_bm25(value: float | str) -> float:
    raw = float(value)
    if raw <= 0:
        return 1.0
    score = 1.0 / (1.0 + raw)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class _BaseSearchAdapter(KnowledgeSearch):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._projection = KnowledgeFTSProjection(connection=connection)
        self._connection = connection

    @staticmethod
    def _current_knowledge_ids(
        connection: sqlite3.Connection,
        memory_space_id: str,
        knowledge_ids: Iterable[str],
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

    def _search_filtered(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
        offset: int,
    ) -> list[SearchHit]:
        if offset < 0 or limit < 0:
            return []

        required = offset + limit
        if required <= 0:
            return []

        fetch_size = max(50, required)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        returned_ids: set[str] = set()

        while len(hits) < required:
            projection_hits = self._projection.search(
                memory_space_id=memory_space_id,
                query=query,
                limit=fetch_size,
            )

            new_hits = []
            for item in projection_hits:
                if item.knowledge_id in seen:
                    continue
                seen.add(item.knowledge_id)
                new_hits.append(item)

            if not new_hits:
                break

            valid_ids = self._current_knowledge_ids(
                self._connection,
                memory_space_id=memory_space_id,
                knowledge_ids=(item.knowledge_id for item in new_hits),
            )

            for item in new_hits:
                if item.knowledge_id in valid_ids and item.knowledge_id not in returned_ids:
                    hits.append(item)
                    returned_ids.add(item.knowledge_id)

            if len(projection_hits) < fetch_size:
                break
            if len(seen) >= required and len(hits) < required:
                fetch_size += required
            else:
                fetch_size *= 2

            if len(hits) >= required:
                break

        return hits[:required]

    def search(self, memory_space_id: str, query: str, limit: int, *, offset: int = 0) -> list[SearchHit]:
        if not query:
            return []
        hits = self._search_filtered(
            memory_space_id=memory_space_id,
            query=query,
            limit=limit,
            offset=offset,
        )
        return hits[offset : offset + limit]


class RawSQLiteKnowledgeSearchAdapter(_BaseSearchAdapter):
    """Search adapter that exposes raw projection ranking."""



class SQLiteKnowledgeSearchAdapter(_BaseSearchAdapter):
    """Search adapter that normalizes BM25 scores into `0..1`."""

    def search(self, memory_space_id: str, query: str, limit: int, *, offset: int = 0) -> list[SearchHit]:
        hits = super().search(
            memory_space_id=memory_space_id,
            query=query,
            limit=limit,
            offset=offset,
        )
        return [
            SearchHit(
                knowledge_id=item.knowledge_id,
                lexical_score=_normalize_bm25(item.lexical_score),
                vector_score=item.vector_score,
            )
            for item in hits
        ]
