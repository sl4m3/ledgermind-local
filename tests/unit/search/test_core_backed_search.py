from __future__ import annotations

from ledgermind_local.core_gateway.contracts import ContextViewResult
from ledgermind_local.core_gateway.search_contracts import SearchHit
from ledgermind_local.search.core_backed import CoreBackedSearch


class FakeLocalSearch:
    def search(self, memory_space_id: str, query: str, limit: int) -> list[SearchHit]:
        assert memory_space_id == "space-1"
        assert query == "query"
        assert limit == 8
        return [
            SearchHit("knowledge-2", lexical_score=-0.1, vector_score=0.9),
            SearchHit("knowledge-1", lexical_score=-0.2, vector_score=0.4),
        ]


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
