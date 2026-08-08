"""Minimal public ContextView endpoint backed by a ContextSearch port."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from ledgermind_local.core_gateway import (
    ContextViewResult,
    DomainRejectedError,
    RecordRetrievalOutcomeV2Command,
    RetrieveContextCommand,
    RetrieveContextV2Command,
    RetrieveContextV2Result,
    TransientCoreError,
)

from .http import build_request_id, error_payload, validate_json_request_headers


class ContextSearch(Protocol):
    """Boundary used by HTTP context routes for Core or Core-backed search."""

    def retrieve_context(self, request: RetrieveContextCommand) -> ContextViewResult: ...

    def retrieve_context_v2(
        self, request: RetrieveContextV2Command
    ) -> RetrieveContextV2Result: ...

    def record_retrieval_outcome_v2(
        self, command: RecordRetrievalOutcomeV2Command
    ) -> None: ...


class QueryEmbedder(Protocol):
    def embed_query(self, memory_space_id: str, query: str) -> Sequence[float]: ...


class ContextRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["1"] = "1"
    memory_space_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=20_000)
    limit: int = Field(default=5, ge=1, le=50)
    query_embedding: list[float] | None = Field(default=None, min_length=1, max_length=8_192)
    embedding_model_id: str | None = Field(default=None, min_length=1, max_length=200)
    embedding_model_version: str | None = Field(default=None, min_length=1, max_length=200)
    project_id: str | None = Field(default=None, min_length=1, max_length=200)
    repository_id: str | None = Field(default=None, min_length=1, max_length=200)
    task_id: str | None = Field(default=None, min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    related_object_ids: list[str] | None = None
    requested_facets: list[str] | None = None
    explanation_level: Literal["compact", "none"] = "compact"


class ContextItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_id: str
    title: str
    target: str
    statement: str
    relevance: float = Field(ge=0.0, le=1.0)
    selection_explanation: dict[str, object] | None = None
    value_id: str | None = None
    primary_object_id: str | None = None
    object_name: str | None = None
    facet: str | None = None
    content: str | None = None


class ContextViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["1", "2"] = "1"
    items: list[ContextItemResponse]
    retrieval_request_id: str | None = None


def create_context_router(
    require_token: Callable[..., str],
    context_search: ContextSearch | None,
    *,
    max_body_bytes: int,
    query_embedder: QueryEmbedder | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/context/retrieve", response_model=ContextViewResponse)
    @router.post("/v1/context/search", response_model=ContextViewResponse)
    def search_context(
        payload: ContextRetrieveRequest,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
    ) -> ContextViewResponse:
        validate_json_request_headers(request.headers, max_bytes=max_body_bytes)
        if context_search is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload("core_unavailable", "Core gateway is unavailable"),
            )
        request_id = build_request_id(request.headers)
        response.headers["X-Request-ID"] = request_id
        try:
            query_embedding = payload.query_embedding
            embedding_model_id = payload.embedding_model_id or "retrieval-embedder"
            embedding_model_version = payload.embedding_model_version or "1"
            if query_embedding is None and query_embedder is not None:
                embed_with_metadata = getattr(query_embedder, "embed_query_with_metadata", None)
                if callable(embed_with_metadata):
                    embedded, embedding_model_id, embedding_model_version = embed_with_metadata(
                        payload.memory_space_id, payload.query
                    )
                    query_embedding = [float(component) for component in embedded]
                else:
                    query_embedding = [
                        float(component)
                        for component in query_embedder.embed_query(
                            payload.memory_space_id, payload.query
                        )
                    ]
            if query_embedding is not None:
                retrieve_v2 = getattr(context_search, "retrieve_context_v2", None)
                if not callable(retrieve_v2):
                    raise TransientCoreError("Core retrieval v2 is unavailable")
                result = retrieve_v2(
                    RetrieveContextV2Command(
                        request_id=request_id,
                        memory_space_id=payload.memory_space_id,
                        query_text=payload.query,
                        query_embedding=tuple(query_embedding),
                        embedding_model_id=embedding_model_id,
                        embedding_model_version=embedding_model_version,
                        limit=payload.limit,
                        project_id=payload.project_id,
                        repository_id=payload.repository_id,
                        task_id=payload.task_id,
                        conversation_id=payload.conversation_id,
                        related_object_ids=tuple(payload.related_object_ids or ()),
                        requested_facets=tuple(payload.requested_facets or ()),
                        explanation_level=payload.explanation_level,
                    )
                )
                response_payload = _context_v2_response(result.payload)
                record_outcome = getattr(
                    context_search, "record_retrieval_outcome_v2", None
                )
                retrieval_request_id = response_payload.get("retrieval_request_id")
                response_items = response_payload.get("items", [])
                if (
                    callable(record_outcome)
                    and isinstance(retrieval_request_id, str)
                    and isinstance(response_items, list)
                    and response_items
                ):
                    candidate_ids = tuple(
                        str(item["knowledge_id"])
                        for item in response_items
                        if isinstance(item, dict)
                        and isinstance(item.get("knowledge_id"), str)
                    )
                    record_outcome(
                        RecordRetrievalOutcomeV2Command(
                            request_id=request_id,
                            retrieval_request_id=retrieval_request_id,
                            candidate_value_ids=candidate_ids,
                            delivered_value_ids=candidate_ids,
                        )
                    )
                return ContextViewResponse.model_validate(response_payload)
            result = context_search.retrieve_context(
                RetrieveContextCommand(
                    request_id=request_id,
                    memory_space_id=payload.memory_space_id,
                    query=payload.query,
                    limit=payload.limit,
                )
            )
            return ContextViewResponse.model_validate(result.to_payload())
        except DomainRejectedError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_payload(exc.code, exc.detail),
            ) from exc
        except TransientCoreError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload("core_unavailable", str(exc)),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_payload("validation_failed", str(exc)),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload(
                    "context_retrieval_failed", "context retrieval failed"
                ),
            ) from exc

    return router


__all__ = [
    "ContextItemResponse",
    "ContextRetrieveRequest",
    "ContextSearch",
    "ContextViewResponse",
    "QueryEmbedder",
    "create_context_router",
]


def _context_v2_response(payload: dict[str, object]) -> dict[str, Any]:
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise TypeError("Core retrieval items must be a list")
    items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise TypeError("Core retrieval item must be an object")
        value_id = raw_item.get("value_id")
        object_name = raw_item.get("object_name")
        facet = raw_item.get("facet")
        content = raw_item.get("content")
        relevance = raw_item.get("relevance")
        if not all(isinstance(value, str) and value for value in (value_id, object_name, facet, content)):
            raise ValueError("Core retrieval item has invalid public fields")
        if not isinstance(relevance, (int, float)):
            raise TypeError("Core retrieval item has invalid relevance")
        items.append(
            {
                "knowledge_id": value_id,
                "title": object_name,
                "target": facet,
                "statement": content,
                "relevance": float(relevance),
                "selection_explanation": raw_item.get("selection_explanation"),
                "value_id": value_id,
                "primary_object_id": raw_item.get("primary_object_id"),
                "object_name": object_name,
                "facet": facet,
                "content": content,
            }
        )
    return {
        "api_version": "2",
        "retrieval_request_id": payload.get("retrieval_request_id"),
        "items": items,
    }
