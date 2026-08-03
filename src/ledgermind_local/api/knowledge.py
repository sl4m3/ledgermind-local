"""Atom and knowledge read endpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from ledgermind_core.application.get_atom import GetAtomQuery
from ledgermind_core.application.get_knowledge import GetKnowledgeQuery
from ledgermind_core.contracts.atom import AtomContent, ExtractionInfo
from ledgermind_core.contracts.common import ContractModel
from ledgermind_core.domain import Atom as DomainAtom
from ledgermind_core.domain import KnowledgeItem as DomainKnowledgeItem
from ledgermind_core.domain import Phase, SourceReference

from ledgermind_local.bootstrap import (
    GetKnowledgeEvidenceQuery,
    GetKnowledgeHistoryQuery,
)

from .http import build_request_id, error_payload


class AtomReadResponse(ContractModel):
    api_version: str = "1"
    atom_id: str
    memory_space_id: str
    content: AtomContent
    extraction: ExtractionInfo
    source: SourceReference
    content_digest: str
    supersedes_atom_id: str | None = None
    created_at: str


class KnowledgeReadResponse(ContractModel):
    api_version: str = "1"
    knowledge_id: str
    memory_space_id: str
    title: str
    target: str
    statement: str
    rationale: str
    phase: str
    version: int
    created_at: str
    updated_at: str
    superseded_by_id: str | None = None
    deleted_at: str | None = None


class KnowledgeHistoryItemResponse(ContractModel):
    revision_id: str
    version: int
    event_type: str
    snapshot: dict[str, object]
    cause_atom_id: str | None = None
    created_at: str


class KnowledgeHistoryResponse(ContractModel):
    api_version: str = "1"
    knowledge_id: str
    memory_space_id: str
    revisions: list[KnowledgeHistoryItemResponse]


class KnowledgeEvidenceItemResponse(ContractModel):
    atom_id: str
    relation: str
    created_at: str


class KnowledgeEvidenceResponse(ContractModel):
    api_version: str = "1"
    knowledge_id: str
    memory_space_id: str
    evidence: list[KnowledgeEvidenceItemResponse]


def _phase_to_text(value: str | Phase) -> str:
    return value.value if isinstance(value, Phase) else str(value)


def _map_atom(atom: DomainAtom) -> AtomReadResponse:
    return AtomReadResponse(
        atom_id=atom.atom_id,
        memory_space_id=atom.memory_space_id,
        content=AtomContent(
            title=atom.content.title,
            target=atom.content.target,
            statement=atom.content.statement,
            rationale=atom.content.rationale,
            result=atom.content.result,
            artifacts=list(atom.content.artifacts),
        ),
        extraction=ExtractionInfo(
            host=atom.extraction.host,
            provider=atom.extraction.provider,
            model=atom.extraction.model,
            prompt_version=atom.extraction.prompt_version,
            schema_version=atom.extraction.schema_version,
            purpose=atom.extraction.purpose,
        ),
        source=SourceReference(
            source_system=atom.source.source_system,
            source_instance_id=atom.source.source_instance_id,
            source_profile_id=atom.source.source_profile_id,
            source_session_id=atom.source.source_session_id,
            source_round_id=atom.source.source_round_id,
            first_message_id=atom.source.first_message_id,
            final_message_id=atom.source.final_message_id,
            message_ids=tuple(atom.source.message_ids),
            source_digest=atom.source.source_digest,
            source_schema_version=atom.source.source_schema_version,
            resolver_version=atom.source.resolver_version,
        ),
        content_digest=atom.content_digest,
        supersedes_atom_id=atom.supersedes_atom_id,
        created_at=atom.created_at.isoformat(),
    )


def _map_knowledge(knowledge: DomainKnowledgeItem) -> KnowledgeReadResponse:
    return KnowledgeReadResponse(
        knowledge_id=knowledge.knowledge_id,
        memory_space_id=knowledge.memory_space_id,
        title=knowledge.title,
        target=knowledge.target,
        statement=knowledge.statement,
        rationale=knowledge.rationale,
        phase=_phase_to_text(knowledge.phase),
        version=knowledge.version,
        created_at=knowledge.created_at.isoformat(),
        updated_at=knowledge.updated_at.isoformat(),
        superseded_by_id=knowledge.superseded_by_id,
        deleted_at=knowledge.deleted_at.isoformat() if knowledge.deleted_at is not None else None,
    )


def _map_history_result(result: Any) -> KnowledgeHistoryResponse:
    return KnowledgeHistoryResponse(
        knowledge_id=result.knowledge_id,
        memory_space_id=result.memory_space_id,
        revisions=[
            KnowledgeHistoryItemResponse(
                revision_id=item.revision_id,
                version=item.version,
                event_type=item.event_type,
                snapshot=item.snapshot,
                cause_atom_id=item.cause_atom_id,
                created_at=item.created_at,
            )
            for item in result.revisions
        ],
    )


def _map_evidence_result(result: Any) -> KnowledgeEvidenceResponse:
    return KnowledgeEvidenceResponse(
        knowledge_id=result.knowledge_id,
        memory_space_id=result.memory_space_id,
        evidence=[
            KnowledgeEvidenceItemResponse(
                atom_id=item.atom_id,
                relation=item.relation,
                created_at=item.created_at,
            )
            for item in result.evidence
        ],
    )


def _resolve_memory_space_id(
    path_space_id: str | None,
    header_space_id: str | None,
) -> str:
    if path_space_id is None and header_space_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_payload(
                "missing_memory_space_id",
                "memory space id is required",
            ),
        )

    if (
        path_space_id is not None
        and header_space_id is not None
        and path_space_id != header_space_id
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_payload(
                "memory_space_mismatch",
                "memory space id in path and header mismatch",
            ),
        )

    if path_space_id is not None:
        return path_space_id
    assert header_space_id is not None
    return header_space_id


def create_knowledge_router(
    require_token: Callable[..., str],
    *,
    get_atom_handler: Any,
    get_knowledge_handler: Any,
    get_knowledge_history_handler: Any,
    get_knowledge_evidence_handler: Any,
) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/v1/memory-spaces/{memory_space_id}/atoms/{atom_id}",
        response_model=AtomReadResponse,
    )
    @router.get("/v1/atoms/{memory_space_id}/{atom_id}", response_model=AtomReadResponse)
    @router.get("/v1/atoms/{atom_id}", response_model=AtomReadResponse)
    def get_atom(
        atom_id: str,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
        memory_space_id: str | None = None,
        memory_space_id_header: str | None = Header(default=None, alias="X-Memory-Space-ID"),
    ) -> AtomReadResponse:
        space_id = _resolve_memory_space_id(
            path_space_id=memory_space_id,
            header_space_id=memory_space_id_header,
        )
        try:
            atom = get_atom_handler.handle(
                GetAtomQuery(memory_space_id=space_id, atom_id=atom_id)
            )
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
                detail=error_payload("knowledge_lookup_failed", "knowledge lookup failed"),
            ) from exc

        if atom is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_payload("not_found", "atom not found"),
            )

        response.headers["X-Request-ID"] = build_request_id(request.headers)
        return _map_atom(atom)

    @router.get(
        "/v1/knowledge/{memory_space_id}/{knowledge_id}",
        response_model=KnowledgeReadResponse,
    )
    @router.get(
        "/v1/memory-spaces/{memory_space_id}/knowledge/{knowledge_id}",
        response_model=KnowledgeReadResponse,
    )
    @router.get("/v1/knowledge/{knowledge_id}", response_model=KnowledgeReadResponse)
    def get_knowledge(
        knowledge_id: str,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
        memory_space_id: str | None = None,
        knowledge_memory_space_id: str | None = Header(
            default=None,
            alias="X-Memory-Space-ID",
        ),
    ) -> KnowledgeReadResponse:
        space_id = _resolve_memory_space_id(
            path_space_id=memory_space_id,
            header_space_id=knowledge_memory_space_id,
        )
        try:
            knowledge = get_knowledge_handler.handle(
                GetKnowledgeQuery(memory_space_id=space_id, knowledge_id=knowledge_id)
            )
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
                detail=error_payload("knowledge_lookup_failed", "knowledge lookup failed"),
            ) from exc

        if knowledge is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_payload("not_found", "knowledge not found"),
            )

        response.headers["X-Request-ID"] = build_request_id(request.headers)
        return _map_knowledge(knowledge)

    @router.get(
        "/v1/memory-spaces/{memory_space_id}/knowledge/{knowledge_id}/history",
        response_model=KnowledgeHistoryResponse,
    )
    def get_knowledge_history(
        memory_space_id: str,
        knowledge_id: str,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
    ) -> KnowledgeHistoryResponse:
        try:
            query = GetKnowledgeHistoryQuery(
                memory_space_id=memory_space_id,
                knowledge_id=knowledge_id,
            )
            result = get_knowledge_history_handler.handle(query)
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
                detail=error_payload("knowledge_history_failed", "knowledge history lookup failed"),
            ) from exc

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_payload("not_found", "knowledge not found"),
            )

        response.headers["X-Request-ID"] = build_request_id(request.headers)
        return _map_history_result(result)

    @router.get(
        "/v1/memory-spaces/{memory_space_id}/knowledge/{knowledge_id}/evidence",
        response_model=KnowledgeEvidenceResponse,
    )
    def get_knowledge_evidence(
        memory_space_id: str,
        knowledge_id: str,
        request: Request,
        response: Response,
        _token: str = Depends(require_token),
    ) -> KnowledgeEvidenceResponse:
        try:
            query = GetKnowledgeEvidenceQuery(
                memory_space_id=memory_space_id,
                knowledge_id=knowledge_id,
            )
            result = get_knowledge_evidence_handler.handle(query)
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
                detail=error_payload(
                    "knowledge_evidence_failed",
                    "knowledge evidence lookup failed",
                ),
            ) from exc

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_payload("not_found", "knowledge not found"),
            )

        response.headers["X-Request-ID"] = build_request_id(request.headers)
        return _map_evidence_result(result)

    return router
