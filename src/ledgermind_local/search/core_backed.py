"""Local projection search followed by Core-owned context reranking."""

from __future__ import annotations

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.contracts import (
    ContextViewResult,
    RetrieveContextCommand,
)
from ledgermind_local.core_gateway.search_contracts import KnowledgeSearch, SearchHit


class CoreBackedSearch:
    """Use Local indexes for candidates and Core for the authoritative context view."""

    def __init__(self, local_search: KnowledgeSearch, core_gateway: CoreGateway) -> None:
        self._local_search = local_search
        self._core_gateway = core_gateway

    def retrieve_context(
        self,
        *,
        request_id: str,
        memory_space_id: str,
        query: str,
        limit: int,
        candidate_limit: int | None = None,
    ) -> ContextViewResult:
        if candidate_limit is None:
            candidate_limit = max(limit * 4, limit)
        hits = self._local_search.search(
            memory_space_id=memory_space_id,
            query=query,
            limit=candidate_limit,
        )
        candidates = tuple(hit.knowledge_id for hit in hits)
        scores = tuple(
            (hit.knowledge_id, _external_score(index, hit, len(hits)))
            for index, hit in enumerate(hits)
        )
        return self._core_gateway.retrieve_context(
            RetrieveContextCommand(
                request_id=request_id,
                memory_space_id=memory_space_id,
                query=query,
                limit=limit,
                candidate_ids=candidates,
                candidate_scores=scores,
            )
        )


def _external_score(index: int, hit: SearchHit, count: int) -> float:
    """Normalize Local lexical/vector evidence into a bounded rerank score."""
    rank_score = 1.0 - (index / max(count, 1))
    vector_score = hit.vector_score if hit.vector_score is not None else rank_score
    return max(0.0, min(1.0, (rank_score + vector_score) / 2.0))


__all__ = ["CoreBackedSearch"]
