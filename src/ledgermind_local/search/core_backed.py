"""Local candidate search followed by Core-owned context reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ledgermind_local.core_gateway.base import CoreGateway
from ledgermind_local.core_gateway.contracts import (
    ContextViewResult,
    RetrieveContextCommand,
)
from ledgermind_local.core_gateway.search_contracts import KnowledgeSearch, SearchHit

DEGRADED_SEARCH_STATUS = "degraded"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One Local projection candidate sent to Core for final ranking."""

    knowledge_id: str
    score: float
    source: str

    def __post_init__(self) -> None:
        knowledge_id = self.knowledge_id.strip()
        source = self.source.strip()
        if not knowledge_id:
            raise ValueError("knowledge_id must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        try:
            score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate score must be numeric") from exc
        if not isfinite(score):
            raise ValueError("candidate score must be finite")
        object.__setattr__(self, "knowledge_id", knowledge_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "score", score)


class LocalCandidateSearch(Protocol):
    """Port implemented by Local FTS/vector candidate projections."""

    def search(
        self,
        memory_space_id: str,
        query: str,
        limit: int,
    ) -> Sequence[CandidateScore]:
        """Return memory-space-scoped candidates with bounded or raw scores."""
        ...


SearchStatusCallback = Callable[[str], None]
CandidateSearchPort = LocalCandidateSearch | KnowledgeSearch


class CoreBackedSearch:
    """Use Local indexes for candidates and Core for the authoritative context view.

    ``local_search`` is the primary candidate source (normally FTS).  An optional
    ``vector_search`` source is queried with the same expanded limit.  Local
    failures are intentionally isolated from Core: the request is retried through
    Core without candidate restrictions and an internal degraded marker is sent to
    ``status_callback``.
    """

    def __init__(
        self,
        local_search: CandidateSearchPort | None = None,
        core_gateway: CoreGateway | None = None,
        *,
        vector_search: LocalCandidateSearch | None = None,
        fts_search: LocalCandidateSearch | None = None,
        status_callback: SearchStatusCallback | None = None,
        candidate_limit: int | None = None,
    ) -> None:
        if local_search is not None and fts_search is not None:
            raise ValueError("local_search and fts_search are mutually exclusive")
        if core_gateway is None:
            raise ValueError("core_gateway is required")
        self._local_search = (
            fts_search if fts_search is not None else local_search
        )
        self._vector_search = vector_search
        self._core_gateway = core_gateway
        self._status_callback = status_callback
        self._candidate_limit = candidate_limit

    def retrieve_context(
        self,
        request: RetrieveContextCommand | None = None,
        *,
        request_id: str | None = None,
        memory_space_id: str | None = None,
        query: str | None = None,
        limit: int | None = None,
        candidate_limit: int | None = None,
    ) -> ContextViewResult:
        if request is not None:
            if any(
                value is not None
                for value in (request_id, memory_space_id, query, limit)
            ):
                raise ValueError("request and keyword retrieval arguments are mutually exclusive")
            request_id = request.request_id
            memory_space_id = request.memory_space_id
            query = request.query
            limit = request.limit
        if request_id is None or memory_space_id is None or query is None or limit is None:
            raise TypeError("request or all retrieval keyword arguments are required")
        local_candidates = self._collect_candidates(
            memory_space_id=memory_space_id,
            query=query,
            limit=limit,
            candidate_limit=candidate_limit,
        )
        candidate_ids: tuple[str, ...] = ()
        candidate_scores: tuple[tuple[str, float], ...] = ()
        if local_candidates is not None:
            candidate_ids, candidate_scores = local_candidates
        return self._core_gateway.retrieve_context(
            RetrieveContextCommand(
                request_id=request_id,
                memory_space_id=memory_space_id,
                query=query,
                limit=limit,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
            )
        )

    def _collect_candidates(
        self,
        *,
        memory_space_id: str,
        query: str,
        limit: int,
        candidate_limit: int | None,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]] | None:
        searches: list[tuple[str, CandidateSearchPort]] = []
        if self._local_search is not None:
            searches.append(("fts", self._local_search))
        if self._vector_search is not None:
            searches.append(("vector", self._vector_search))
        if not searches:
            self._mark_degraded()
            return None

        expanded_limit = self._expanded_limit(limit, candidate_limit)
        evidence: list[CandidateScore] = []
        try:
            for default_source, search in searches:
                raw_candidates = search.search(
                    memory_space_id=memory_space_id,
                    query=query,
                    limit=expanded_limit,
                )
                materialized = list(raw_candidates)
                for index, candidate in enumerate(materialized):
                    evidence.extend(
                        _coerce_candidates(
                            candidate,
                            default_source=default_source,
                            index=index,
                            count=len(materialized),
                        )
                    )
        except Exception:  # noqa: BLE001 - Local projection failures trigger Core fallback
            self._mark_degraded()
            return None

        merged = _merge_candidates(evidence)
        candidate_cap = max(limit * 10, 100)
        merged = merged[:candidate_cap]
        candidates = tuple(candidate_id for candidate_id, _ in merged)
        scores = tuple(merged)
        return candidates, scores

    def _expanded_limit(self, limit: int, candidate_limit: int | None) -> int:
        configured = candidate_limit
        if configured is None:
            configured = self._candidate_limit
        if configured is None:
            configured = max(limit * 4, limit)
        return max(int(configured), limit)

    def _mark_degraded(self) -> None:
        if self._status_callback is None:
            return
        try:
            self._status_callback(DEGRADED_SEARCH_STATUS)
        except Exception:  # noqa: BLE001 - status sinks must not affect retrieval
            # A health/status sink must never turn a Core fallback into a 503.
            return


def _coerce_candidates(
    candidate: object,
    *,
    default_source: str,
    index: int,
    count: int,
) -> tuple[CandidateScore, ...]:
    if isinstance(candidate, CandidateScore):
        return (candidate,)
    if isinstance(candidate, SearchHit):
        lexical = CandidateScore(
            knowledge_id=candidate.knowledge_id,
            score=_rank_score(index, count),
            source=default_source,
        )
        if candidate.vector_score is None:
            return (lexical,)
        vector = CandidateScore(
            knowledge_id=candidate.knowledge_id,
            score=_bounded_score(candidate.vector_score),
            source="vector",
        )
        return lexical, vector
    raise TypeError("local search returned an unsupported candidate type")


def _merge_candidates(
    candidates: Sequence[CandidateScore],
) -> list[tuple[str, float]]:
    by_id: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        source_scores = by_id.setdefault(candidate.knowledge_id, {})
        bounded = _bounded_score(candidate.score)
        source_scores[candidate.source] = max(
            bounded,
            source_scores.get(candidate.source, 0.0),
        )

    merged = [
        (
            knowledge_id,
            round(sum(source_scores.values()) / len(source_scores), 12),
        )
        for knowledge_id, source_scores in by_id.items()
    ]
    merged.sort(key=lambda item: (-item[1], item[0]))
    return merged


def _rank_score(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return _bounded_score(1.0 - (index / count))


def _bounded_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate score must be numeric") from exc
    if not isfinite(score):
        raise ValueError("candidate score must be finite")
    return max(0.0, min(1.0, score))


__all__ = [
    "DEGRADED_SEARCH_STATUS",
    "CandidateScore",
    "CoreBackedSearch",
    "LocalCandidateSearch",
    "SearchStatusCallback",
]
