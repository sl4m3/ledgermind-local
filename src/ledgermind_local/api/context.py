"""Minimal public ContextView endpoint backed by a ContextSearch port."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from ledgermind_local.core_gateway import (
    ContextViewResult,
    DomainRejectedError,
    RetrieveContextCommand,
    TransientCoreError,
)

from .http import build_request_id, error_payload, validate_json_request_headers


class ContextSearch(Protocol):
    """Boundary used by HTTP context routes for Core or Core-backed search."""

    def retrieve_context(self, request: RetrieveContextCommand) -> ContextViewResult: ...


class ContextRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["1"] = "1"
    memory_space_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=20_000)
    limit: int = Field(default=5, ge=1, le=50)


class ContextItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_id: str
    title: str
    target: str
    statement: str
    relevance: float = Field(ge=0.0, le=1.0)


class ContextViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["1"] = "1"
    items: list[ContextItemResponse]


def create_context_router(
    require_token: Callable[..., str],
    context_search: ContextSearch | None,
    *,
    max_body_bytes: int,
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
    "create_context_router",
]
