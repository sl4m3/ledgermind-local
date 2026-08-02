"""Atom ingestion HTTP endpoint."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from starlette.requests import Headers

from application import IdempotencyConflict, UnsupportedEvolutionDecision
from application.mappers import IngestAtomCommand
from application import calculate_request_hash
from contracts import IngestAtomRequest, IngestAtomResult
from domain import AtomContent, ExtractionInfo, SourceReference

from .dependencies import Settings


_DEFAULT_BODY_LIMIT_BYTES = 128 * 1024


def _enforce_body_limit(headers: Headers, *, raw: bytes) -> None:
    length = headers.get("content-length")
    if length is not None:
        try:
            if int(length) > _DEFAULT_BODY_LIMIT_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="payload too large",
                )
        except ValueError:
            pass

    if len(raw) > _DEFAULT_BODY_LIMIT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="payload too large",
        )


def _build_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def _normalize_text(value: str) -> str:
    return value.strip()


def _build_command(payload: IngestAtomRequest) -> IngestAtomCommand:
    return IngestAtomCommand(
        idempotency_key=payload.idempotency_key,
        request_hash=calculate_request_hash(payload.model_dump()),
        memory_space_id=_normalize_text(payload.memory_space_id),
        source=SourceReference(
            source_system=payload.source.source_system,
            source_instance_id=_normalize_text(payload.source.source_instance_id),
            source_profile_id=_normalize_text(payload.source.source_profile_id),
            source_session_id=_normalize_text(payload.source.source_session_id),
            source_round_id=_normalize_text(payload.source.source_round_id),
            first_message_id=(
                _normalize_text(payload.source.first_message_id)
                if payload.source.first_message_id is not None
                else None
            ),
            final_message_id=(
                _normalize_text(payload.source.final_message_id)
                if payload.source.final_message_id is not None
                else None
            ),
            message_ids=tuple(
                _normalize_text(message_id) for message_id in payload.source.message_ids
            ),
            source_digest=payload.source.source_digest,
            source_schema_version=payload.source.source_schema_version,
            resolver_version=payload.source.resolver_version,
        ),
        content=AtomContent(
            title=_normalize_text(payload.atom.title),
            target=_normalize_text(payload.atom.target),
            statement=_normalize_text(payload.atom.statement),
            rationale=_normalize_text(payload.atom.rationale),
            result=_normalize_text(payload.atom.result),
            artifacts=tuple(_normalize_text(artifact) for artifact in payload.atom.artifacts),
        ),
        extraction=ExtractionInfo(
            host=_normalize_text(payload.extraction.host),
            provider=_normalize_text(payload.extraction.provider),
            model=_normalize_text(payload.extraction.model),
            prompt_version=payload.extraction.prompt_version,
            schema_version=payload.extraction.schema_version,
            purpose=_normalize_text(payload.extraction.purpose),
        ),
    )


def create_atoms_router(
    require_token,
    ingest_handler: Any,
    *,
    settings: Settings | None = None,
) -> APIRouter:
    router = APIRouter()
    _ = settings

    @router.post("/v1/atoms", status_code=201)
    async def ingest_atom(
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
    ) -> IngestAtomResult:
        raw = await request.body()
        _enforce_body_limit(request.headers, raw=raw)

        try:
            payload = IngestAtomRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid request payload",
            ) from exc

        try:
            command = _build_command(payload)
            result = ingest_handler.handle(command)
        except IdempotencyConflict as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"idempotency conflict: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except UnsupportedEvolutionDecision as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
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
                detail="ingestion failed",
            ) from exc

        response.status_code = status.HTTP_201_CREATED if not result.duplicate else status.HTTP_200_OK
        response.headers["X-Request-ID"] = _build_request_id(request)
        response.headers["Cache-Control"] = "no-store"
        return IngestAtomResult(
            api_version="1",
            atom_id=result.atom_id,
            knowledge_id=result.knowledge_id,
            knowledge_version=result.knowledge_version,
            phase=result.phase,
            duplicate=result.duplicate,
            projections_pending=result.projections_pending,
        )

    return router
