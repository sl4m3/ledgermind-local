"""Context retrieval HTTP endpoint."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from application import map_context_query
from contracts import RetrieveContextRequest, RetrieveContextResult


def create_context_router(
    require_token,
    retrieve_context_handler,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/context/search",
        response_model=RetrieveContextResult,
    )
    async def search_context(
        request: Request,
        _token: str = Depends(require_token),
    ) -> RetrieveContextResult:
        raw = await request.body()
        try:
            payload = RetrieveContextRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid request payload",
            ) from exc

        try:
            query = map_context_query(payload.model_dump())
            return retrieve_context_handler.handle(query)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except sqlite3.DatabaseError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="context retrieval failed",
            ) from exc

    return router
