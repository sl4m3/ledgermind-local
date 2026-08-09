from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ledgermind_local.api.context import create_context_router
from ledgermind_local.core_gateway.contracts import (
    RecordRetrievalOutcomeCommand,
    RetrieveContextCommand,
    RetrieveContextResult,
)


class _Gateway:
    def __init__(self) -> None:
        self.request: RetrieveContextCommand | None = None
        self.outcomes: list[RecordRetrievalOutcomeCommand] = []

    def retrieve_context(
        self, request: RetrieveContextCommand
    ) -> RetrieveContextResult:
        self.request = request
        return RetrieveContextResult(
            {
                "retrieval_request_id": "retrieval-1",
                "items": [
                    {
                        "value_id": "value-1",
                        "primary_object_id": "object-1",
                        "object_name": "Deployment",
                        "facet": "property",
                        "content": "Deployments require review.",
                        "relevance": 0.9,
                        "explanation": {
                            "object_reasons": ["direct_value_semantic"],
                            "item_facet": "property",
                            "activated_facets": [],
                            "score_components": {
                                "semantic": 0.9,
                                "object": 0.0,
                                "facet": 0.0,
                                "scope_time": 0.0,
                                "context": 0.0,
                                "recency": 0.0,
                                "support": 0.0,
                                "usage": 0.0,
                            },
                        },
                    }
                ],
            }
        )

    def record_retrieval_outcome(
        self, command: RecordRetrievalOutcomeCommand
    ) -> None:
        self.outcomes.append(command)


class _Embedder:
    def embed_query_with_metadata(
        self, memory_space_id: str, query: str
    ) -> tuple[tuple[float, ...], str, str]:
        assert memory_space_id == "space"
        assert query == "deployment"
        return (0.1, 0.2), "embedder", "2026-08"


def test_context_embeds_query_returns_provenance_and_records_outcome() -> None:
    gateway = _Gateway()
    app = FastAPI()
    app.include_router(
        create_context_router(
            lambda: "token",
            gateway,
            max_body_bytes=100_000,
            query_embedder=_Embedder(),
        )
    )

    response = TestClient(app).post(
        "/context/retrieve",
        headers={"X-Request-ID": "request-1"},
        json={
            "memory_space_id": "space",
            "query": "deployment",
            "project_id": "project-1",
            "repository_id": "repository-1",
            "task_id": "task-1",
            "conversation_id": "conversation-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 2
    assert payload["retrieval_request_id"] == "retrieval-1"
    assert payload["delivered_value_ids"] == ["value-1"]
    assert payload["items"][0]["object_name"] == "Deployment"
    assert payload["items"][0]["facet"] == "property"
    assert payload["items"][0]["explanation"]["object_reasons"] == [
        "direct_value_semantic"
    ]
    assert gateway.request is not None
    assert gateway.request.query_embedding == (0.1, 0.2)
    assert gateway.request.embedding_model_id == "embedder"
    assert gateway.request.repository_id == "repository-1"
    assert gateway.outcomes[0].candidate_value_ids == ("value-1",)
    assert gateway.outcomes[0].delivered_value_ids == ("value-1",)
