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
            raise HTTPException(
                status_code=503, detail="Control maintenance is unavailable"
            )
        try:
            result = run()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail="Control maintenance failed"
            ) from exc
        if not isinstance(result, dict):
            raise HTTPException(
                status_code=503, detail="Control maintenance result is malformed"
            )
        if result.get("status") != "completed" or result.get("ready") is not True:
            raise HTTPException(
                status_code=503, detail="Control maintenance is not ready"
            )
        return result

    @router.post("/replay-failed")
    def replay_failed_user_semantic(
        limit: int = 100,
        _token: str = Depends(require_token),
    ) -> dict[str, Any]:
        del _token
        if runtime is None:
            raise HTTPException(status_code=503, detail="Local runtime is unavailable")
        replay = getattr(runtime, "retry_failed_user_semantic", None)
        if not callable(replay):
            raise HTTPException(status_code=503, detail="Replay is unavailable")
        if not 1 <= limit <= 1_000:
            raise HTTPException(
                status_code=422, detail="limit must be between 1 and 1000"
            )
        try:
            result = replay(limit=limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Replay failed") from exc
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise HTTPException(status_code=503, detail="Replay result is malformed")
        return result

    return router


__all__ = ["create_maintenance_router"]
