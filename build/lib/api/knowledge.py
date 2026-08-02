"""Atom and knowledge read endpoints."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from application.get_atom import GetAtomHandler, GetAtomQuery
from application.get_knowledge import GetKnowledgeHandler, GetKnowledgeQuery
from contracts.atom import AtomContent, ExtractionInfo, IngestAtomRequest, SourceReference
from contracts.common import ContractModel
from domain import Phase
from domain import Atom as DomainAtom
from domain import KnowledgeItem as DomainKnowledgeItem


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
            message_ids=list(atom.source.message_ids),
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


def create_knowledge_router(
    require_token,
    *,
    get_atom_handler: Any,
    get_knowledge_handler: Any,
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/atoms/{memory_space_id}/{atom_id}", response_model=AtomReadResponse)
    async def get_atom(
        memory_space_id: str,
        atom_id: str,
        _token: str = Depends(require_token),
    ) -> AtomReadResponse:
        try:
            atom = get_atom_handler.handle(
                GetAtomQuery(memory_space_id=memory_space_id, atom_id=atom_id)
            )
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
                detail="knowledge lookup failed",
            ) from exc

        if atom is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="atom not found",
            )

        return _map_atom(atom)

    @router.get(
        "/v1/knowledge/{memory_space_id}/{knowledge_id}",
        response_model=KnowledgeReadResponse,
    )
    async def get_knowledge(
        memory_space_id: str,
        knowledge_id: str,
        _token: str = Depends(require_token),
    ) -> KnowledgeReadResponse:
        try:
            knowledge = get_knowledge_handler.handle(
                GetKnowledgeQuery(memory_space_id=memory_space_id, knowledge_id=knowledge_id)
            )
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
                detail="knowledge lookup failed",
            ) from exc

        if knowledge is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="knowledge not found",
            )

        return _map_knowledge(knowledge)

    return router
