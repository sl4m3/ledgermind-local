"""Context retrieval HTTP endpoint."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from ledgermind_core.application import map_context_query
from ledgermind_core.contracts import RetrieveContextRequest, RetrieveContextResult
from pydantic import ValidationError

from .http import build_request_id, error_payload, validate_json_request


def create_context_router(
    require_token: object,
    retrieve_context_handler: object,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/context/search",
        response_model=RetrieveContextResult,
    )
    async def search_context(
        request: Request,
        response: Response,
        _token: str = Depends(require_token),  # type: ignore[arg-type]
    ) -> RetrieveContextResult:
        raw = await request.body()
        validate_json_request(request.headers, raw=raw)
        try:
            payload = RetrieveContextRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_payload("invalid_request_payload", "invalid request payload"),
            ) from exc

        try:
            query = map_context_query(payload.model_dump())
            response.headers["X-Request-ID"] = build_request_id(request.headers)
            return retrieve_context_handler.handle(query)  # type: ignore[attr-defined,no-any-return]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_payload("validation_failed", str(exc)),
            ) from exc
        except sqlite3.DatabaseError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_payload("database_unavailable", "database unavailable"),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_payload("context_retrieval_failed", "context retrieval failed"),
            ) from exc

    return router
