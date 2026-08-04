"""FTS candidate adapter for the event-derived Core knowledge projection."""

from __future__ import annotations

import sqlite3

from ledgermind_local.core_gateway.search_contracts import SearchHit
from ledgermind_local.projections.fts import KnowledgeFTSProjection

from .core_backed import CandidateScore, LocalCandidateSearch

__all__ = ["CoreProjectionSearchAdapter"]
_FTS_TABLE = "core_knowledge_fts"


class CoreProjectionSearchAdapter(LocalCandidateSearch):
    """Return memory-space-scoped FTS candidates in the A4 candidate contract."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._projection = KnowledgeFTSProjection(connection=connection)

    def search(self, memory_space_id: str, query: str, limit: int) -> list[CandidateScore]:
        if not self._projection_available():
            raise RuntimeError("Core FTS projection is unavailable")
        hits = self._projection.search_core(memory_space_id, query, limit)
        scores = _lexical_scores(hits)
        return [
            CandidateScore(
                knowledge_id=hit.knowledge_id,
                score=score,
                source="fts",
            )
            for hit, score in zip(hits, scores, strict=True)
        ]

    def _projection_available(self) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ? LIMIT 1
            """,
            (_FTS_TABLE,),
        ).fetchone()
        return row is not None


def _lexical_scores(hits: list[SearchHit]) -> list[float]:
    """Convert SQLite's lower-is-better BM25 score to a bounded candidate score."""

    if len(hits) <= 1:
        return [1.0] * len(hits)
    raw_scores = [float(hit.lexical_score) for hit in hits]
    best = min(raw_scores)
    worst = max(raw_scores)
    if best == worst:
        return [
            max(0.0, min(1.0, 1.0 - (index / len(hits))))
            for index, _ in enumerate(hits)
        ]
    span = worst - best
    return [max(0.0, min(1.0, (worst - score) / span)) for score in raw_scores]
