"""Health endpoints for the local LedgerMind service."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette import status

from diagnostics.health import run_readiness_checks


def create_health_router(
    require_token: object,
    maybe_token: object,
    *,
    database_path: Path,
    service_lock_path: Path | None,
    write_handler: object,
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/health/live")
    @router.get("/health/live")
    def health_live(details: bool = Depends(maybe_token)) -> dict[str, str | bool]:  # type: ignore[arg-type]
        payload: dict[str, str | bool] = {"status": "ok"}
        if details:
            payload["healthy"] = True
            payload["database_path"] = str(database_path)
        return payload

    @router.get("/v1/health/ready", response_model=None)
    def health_ready(_token: str = Depends(require_token)) -> object:  # type: ignore[arg-type]
        del _token
        report = run_readiness_checks(
            database_path=database_path,
            service_lock_path=service_lock_path,
            write_handler=write_handler,
        )
        if report["ready"]:
            return report

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=report,
        )

    @router.get("/v1/health/details")
    @router.get("/health/details")
    def health_details(_token: str = Depends(require_token)) -> object:  # type: ignore[arg-type]
        del _token
        report = run_readiness_checks(
            database_path=database_path,
            service_lock_path=service_lock_path,
            write_handler=write_handler,
        )
        if report["ready"]:
            return report

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=report,
        )

    return router
