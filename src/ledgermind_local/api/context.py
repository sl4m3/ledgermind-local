"""Public ContextView endpoint backed directly by the Core gateway."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from ledgermind_local.core_gateway import (
    DomainRejectedError,
    RecordRetrievalOutcomeCommand,
    RetrieveContextCommand,
    RetrieveContextResult,
    TransientCoreError,
)

from .http import build_request_id, error_payload, validate_json_request_headers


class ContextGateway(Protocol):
    """Core context boundary used by the public retrieval route."""

    def retrieve_context(
        self, request: RetrieveContextCommand
    ) -> RetrieveContextResult: ...

    def record_retrieval_outcome(
        self, command: RecordRetrievalOutcomeCommand
    ) -> None: ...


class QueryEmbedder(Protocol):
    def embed_query(self, memory_space_id: str, query: str) -> Sequence[float]: ...


class ContextRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
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

    value_id: str
    primary_object_id: str
    object_name: str
    facet: str
    content: str
    relevance: float = Field(ge=0.0, le=1.0)
    explanation: dict[str, object]


class ContextViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    items: list[ContextItemResponse]
    retrieval_request_id: str
    delivered_value_ids: list[str] = Field(default_factory=list)


def create_context_router(
    require_token: Callable[..., str],
    context_gateway: ContextGateway | None,
    *,
    max_body_bytes: int,
    query_embedder: QueryEmbedder | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/context/retrieve", response_model=ContextViewResponse)
    def retrieve_context(
        payload: ContextRetrieveRequest,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
    ) -> ContextViewResponse:
        validate_json_request_headers(request.headers, max_bytes=max_body_bytes)
        if context_gateway is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload("core_unavailable", "Core gateway is unavailable"),
            )
        request_id = build_request_id(request.headers)
        response.headers["X-Request-ID"] = request_id
        try:
            embedding_model_id = payload.embedding_model_id or "retrieval-embedder"
            embedding_model_version = payload.embedding_model_version or "1"
            query_embedding: list[float]
            if query_embedder is not None:
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
            else:
                if payload.query_embedding is None:
                    raise TransientCoreError("embedding profile is unavailable")
                query_embedding = payload.query_embedding
            retrieve = getattr(context_gateway, "retrieve_context", None)
            if not callable(retrieve):
                raise TransientCoreError("Core retrieval is unavailable")
            result = retrieve(
                RetrieveContextCommand(
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
            response_payload = _context_response(result.payload)
            record_outcome = getattr(
                context_gateway, "record_retrieval_outcome", None
            )
            retrieval_request_id = response_payload["retrieval_request_id"]
            response_items = response_payload["items"]
            candidate_ids = tuple(str(item["value_id"]) for item in response_items)
            if callable(record_outcome) and candidate_ids:
                # Core returns the authoritative candidate set.  The HTTP
                # response represents all returned candidates as delivered;
                # the two lists remain separate in the durable outcome event.
                record_outcome(
                    RecordRetrievalOutcomeCommand(
                        request_id=request_id,
                        retrieval_request_id=retrieval_request_id,
                        candidate_value_ids=candidate_ids,
                        delivered_value_ids=candidate_ids,
                    )
                )
            response_payload["delivered_value_ids"] = list(candidate_ids)
            return ContextViewResponse.model_validate(response_payload)
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
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload("core_unavailable", str(exc)),
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
    "ContextGateway",
    "ContextItemResponse",
    "ContextRetrieveRequest",
    "ContextViewResponse",
    "QueryEmbedder",
    "create_context_router",
]


def _context_response(payload: dict[str, object]) -> dict[str, Any]:
    try:
        from ledgermind_protocol.object_facet import RetrievalResponse

        response = RetrievalResponse.model_validate(payload)
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError("Core retrieval response is not a strict object-facet response") from exc
    items = [item.model_dump(mode="json") for item in response.items]
    return {
        "schema_version": 2,
        "retrieval_request_id": response.retrieval_request_id,
        "items": items,
    }
