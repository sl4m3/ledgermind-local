from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from ledgermind_local.core_gateway.contracts import ContextViewResult
from ledgermind_local.core_gateway.search_contracts import SearchHit
from ledgermind_local.persistence import rounds_migrations
from ledgermind_local.projections import VectorProjectionStore
from ledgermind_local.search.core_backed import (
    CandidateScore,
    CoreBackedSearch,
)
from ledgermind_local.search.fts import CoreProjectionSearchAdapter
from ledgermind_local.search.vector import CoreProjectionVectorSearchAdapter


class FakeLocalSearch:
    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        assert memory_space_id == "space-1"
        assert query == "query"
        assert limit == 8
        return [
            SearchHit("knowledge-2", lexical_score=-0.1, vector_score=0.9),
            SearchHit("knowledge-1", lexical_score=-0.2, vector_score=0.4),
        ]


class StaticCandidateSearch:
    def __init__(self, candidates: list[CandidateScore]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, str, int]] = []

    def search(
        self, memory_space_id: str, query: str, limit: int
    ) -> list[CandidateScore]:
        self.calls.append((memory_space_id, query, limit))
        return self.candidates[:limit]


class FailingLocalSearch:
    def search(
        self, memory_space_id: str, query: str, limit: int
    ) -> list[CandidateScore]:
        raise RuntimeError("local projection is unavailable")


class FakeCoreGateway:
    def __init__(self) -> None:
        self.command = None

    def retrieve_context(self, command):
        self.command = command
        return ContextViewResult(
            items=(),
            api_version="1",
        )


def test_local_candidates_are_sent_to_core_without_database_access() -> None:
    gateway = FakeCoreGateway()
    search = CoreBackedSearch(FakeLocalSearch(), gateway)  # type: ignore[arg-type]

    result = search.retrieve_context(
        request_id="request-1",
        memory_space_id="space-1",
        query="query",
        limit=2,
    )

    assert result.api_version == "1"
    assert gateway.command is not None
    assert gateway.command.candidate_ids == ("knowledge-2", "knowledge-1")
    assert tuple(item_id for item_id, _ in gateway.command.candidate_scores) == (
        "knowledge-2",
        "knowledge-1",
    )


def test_fts_and_vector_candidates_are_merged_and_deduplicated() -> None:
    fts = StaticCandidateSearch(
        [
            CandidateScore("shared", 0.8, "fts"),
            CandidateScore("fts-only", 0.5, "fts"),
        ]
    )
    vector = StaticCandidateSearch(
        [
            CandidateScore("shared", 0.4, "vector"),
            CandidateScore("vector-only", 0.95, "vector"),
        ]
    )
    gateway = FakeCoreGateway()
    search = CoreBackedSearch(
        fts,
        gateway,
        vector_search=vector,
    )

    search.retrieve_context(
        request_id="request-hybrid",
        memory_space_id="space-1",
        query="query",
        limit=2,
    )

    assert fts.calls == [("space-1", "query", 8)]
    assert vector.calls == [("space-1", "query", 8)]
    assert gateway.command is not None
    assert gateway.command.candidate_ids == (
        "vector-only",
        "shared",
        "fts-only",
    )
    assert gateway.command.candidate_scores == (
        ("vector-only", 0.95),
        ("shared", 0.6),
        ("fts-only", 0.5),
    )


def test_candidate_limit_is_bounded_before_core_call() -> None:
    local = StaticCandidateSearch(
        [CandidateScore(f"knowledge-{index}", 1.0, "fts") for index in range(150)]
    )
    gateway = FakeCoreGateway()
    search = CoreBackedSearch(local, gateway)

    search.retrieve_context(
        request_id="request-limit",
        memory_space_id="space-1",
        query="query",
        limit=1,
        candidate_limit=150,
    )

    assert gateway.command is not None
    assert len(gateway.command.candidate_ids) == 100


def test_local_search_failure_falls_back_to_core_and_marks_degraded() -> None:
    gateway = FakeCoreGateway()
    statuses: list[str] = []
    search = CoreBackedSearch(
        FailingLocalSearch(),
        gateway,
        status_callback=statuses.append,
    )

    result = search.retrieve_context(
        request_id="request-fallback",
        memory_space_id="space-1",
        query="query",
        limit=2,
    )

    assert result.api_version == "1"
    assert gateway.command is not None
    assert gateway.command.candidate_ids == ()
    assert gateway.command.candidate_scores == ()
    assert statuses == ["degraded"]


def test_unavailable_fts_projection_falls_back_to_core_and_marks_degraded() -> None:
    connection = sqlite3.connect(":memory:")
    gateway = FakeCoreGateway()
    statuses: list[str] = []
    search = CoreBackedSearch(
        CoreProjectionSearchAdapter(connection),
        gateway,  # type: ignore[arg-type]
        status_callback=statuses.append,
    )

    result = search.retrieve_context(
        request_id="request-fts-fallback",
        memory_space_id="space-1",
        query="query",
        limit=2,
    )

    assert result.api_version == "1"
    assert gateway.command is not None
    assert gateway.command.candidate_ids == ()
    assert gateway.command.candidate_scores == ()
    assert statuses == ["degraded"]
    connection.close()


def test_fts_search_excludes_a_different_memory_space() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    rounds_migrations.apply_migrations(connection)
    connection.executemany(
        """
        INSERT INTO core_knowledge_fts
            (knowledge_id, memory_space_id, title, target, statement)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("space-1-knowledge", "space-1", "shared query", "target", "statement"),
            ("space-2-knowledge", "space-2", "shared query", "target", "statement"),
        ],
    )

    hits = CoreProjectionSearchAdapter(connection).search(
        "space-1", "shared query", limit=10
    )

    assert [hit.knowledge_id for hit in hits] == ["space-1-knowledge"]
    assert hits[0].source == "fts"
    assert hits[0].score == 1.0
    connection.close()


class FakeVectorizer:
    @property
    def dimension(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        return "test"

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]

    def close(self) -> None:
        return None


def test_vector_search_is_scoped_by_fts_memory_space(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    rounds_migrations.apply_migrations(connection)
    connection.executemany(
        """
        INSERT INTO core_knowledge_fts
            (knowledge_id, memory_space_id, title, target, statement)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("space-1-vector", "space-1", "title", "target", "statement"),
            ("space-2-vector", "space-2", "title", "target", "statement"),
        ],
    )
    store = VectorProjectionStore(tmp_path / "vectors", model_dimension=2)
    store.rebuild(
        ["space-1-vector", "space-2-vector"],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    adapter = CoreProjectionVectorSearchAdapter(
        connection=connection,
        vector_store_root=tmp_path / "vectors",
        vectorizer_factory=FakeVectorizer,
    )
    candidates = adapter.search("space-1", "query", limit=10)

    assert [candidate.knowledge_id for candidate in candidates] == ["space-1-vector"]
    assert candidates[0].source == "vector"
    assert candidates[0].score == 1.0
    adapter.close()
    connection.close()
