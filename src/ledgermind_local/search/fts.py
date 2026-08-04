"""Search adapter for the Local projection of Rust Core knowledge."""

from __future__ import annotations

import sqlite3

from ledgermind_local.core_gateway.search_contracts import KnowledgeSearch, SearchHit
from ledgermind_local.projections.fts import KnowledgeFTSProjection

__all__ = ["CoreProjectionSearchAdapter"]


class CoreProjectionSearchAdapter(KnowledgeSearch):
    """Search adapter that reads only the event-derived Core FTS projection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._projection = KnowledgeFTSProjection(connection=connection)

    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        return self._projection.search_core(memory_space_id, query, limit)