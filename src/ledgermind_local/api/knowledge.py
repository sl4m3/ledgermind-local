"""Read-only knowledge export for an authenticated runtime client."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


class KnowledgeExportRequest(BaseModel):
    memory_space_prefix: str = Field(min_length=1, max_length=500)


def create_knowledge_router(
    require_token: Callable[..., str],
    core_gateway: object | None,
    *,
    database_path: Path,
) -> APIRouter:
    router = APIRouter(prefix="/knowledge", tags=["knowledge"])

    @router.post("/export")
    def export_knowledge(
        payload: KnowledgeExportRequest,
        _token: str = Depends(require_token),
    ) -> dict[str, Any]:
        del _token
        prefix = payload.memory_space_prefix.strip()
        if not prefix:
            raise HTTPException(status_code=422, detail="memory_space_prefix is required")
        export_snapshot = getattr(core_gateway, "get_object_facet_snapshot", None)
        if not callable(export_snapshot):
            raise HTTPException(status_code=503, detail="Knowledge export is unavailable")

        try:
            with sqlite3.connect(database_path) as connection:
                memory_space_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT memory_space_id FROM memory_spaces ORDER BY memory_space_id"
                    )
                    if str(row[0]).startswith(prefix)
                ]
            spaces = [
                {
                    "memory_space_id": memory_space_id.removeprefix(prefix) or "default",
                    "snapshot": export_snapshot(memory_space_id, uuid4().hex),
                }
                for memory_space_id in memory_space_ids
            ]
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Knowledge export failed") from exc

        return {
            "schema_version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memory_spaces": spaces,
        }

    @router.post("/purge")
    def purge_knowledge(
        payload: KnowledgeExportRequest,
        _token: str = Depends(require_token),
    ) -> dict[str, Any]:
        del _token
        prefix = payload.memory_space_prefix.strip()
        if not prefix:
            raise HTTPException(status_code=422, detail="memory_space_prefix is required")
        purge_memory_space = getattr(core_gateway, "delete_memory_space", None)
        if not callable(purge_memory_space):
            raise HTTPException(status_code=503, detail="Knowledge deletion is unavailable")

        try:
            with sqlite3.connect(database_path) as connection:
                memory_space_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT memory_space_id FROM memory_spaces ORDER BY memory_space_id"
                    )
                    if str(row[0]).startswith(prefix)
                ]
            for memory_space_id in memory_space_ids:
                purge_memory_space(memory_space_id, uuid4().hex)
            with sqlite3.connect(database_path) as connection:
                connection.executemany(
                    "DELETE FROM memory_spaces WHERE memory_space_id = ?",
                    [(memory_space_id,) for memory_space_id in memory_space_ids],
                )
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Knowledge deletion failed") from exc

        return {"deleted_memory_spaces": len(memory_space_ids)}

    return router


__all__ = ["create_knowledge_router"]
