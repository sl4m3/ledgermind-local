"""Authenticated Local maintenance endpoints backed by the Core gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException


def create_maintenance_router(
    require_token: Callable[..., str], runtime: object | None
) -> APIRouter:
    router = APIRouter(prefix="/maintenance", tags=["maintenance"])

    @router.post("/control")
    def run_control_maintenance(
        _token: str = Depends(require_token),
    ) -> dict[str, Any]:
        del _token
        if runtime is None:
            raise HTTPException(status_code=503, detail="Local runtime is unavailable")
        run = getattr(runtime, "run_control_maintenance", None)
        if not callable(run):
            raise HTTPException(status_code=503, detail="Control maintenance is unavailable")
        try:
            result = run()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Control maintenance failed") from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=503, detail="Control maintenance result is malformed")
        if result.get("status") != "completed" or result.get("ready") is not True:
            raise HTTPException(status_code=503, detail="Control maintenance is not ready")
        return result

    return router


__all__ = ["create_maintenance_router"]
