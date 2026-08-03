"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from ledgermind_local.bootstrap import (
    build_get_atom_handler,
    build_get_knowledge_evidence_handler,
    build_get_knowledge_handler,
    build_get_knowledge_history_handler,
    build_ingest_atom_handler,
    build_ingest_raw_round_handler,
    build_retrieve_context_handler,
)

from .atoms import create_atoms_router
from .auth import (
    build_bearer_token_dependency,
    build_optional_bearer_token_dependency,
)
from .context import create_context_router
from .dependencies import Application, Settings
from .errors import AuthenticationError, authentication_error_handler
from .health import create_health_router
from .http import MAX_JSON_BODY_BYTES, error_payload
from .knowledge import create_knowledge_router
from .rounds import create_rounds_router


def _normalize_database_path(database_path: str | Path) -> Path:
    return Path(database_path)


def create_app(
    application: Application,
    settings: Settings,
    *,
    projection_names: tuple[str, ...] | None = None,
) -> FastAPI:
    """Create a fast API application object using explicit dependencies.

    The function never creates or mutates database files on its own.
    """

    app = FastAPI()
    app.state.application = application
    app.state.database_path = _normalize_database_path(settings.database_path)

    @app.middleware("http")
    async def reject_oversized_json(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        content_length = request.headers.get("content-length")
        if content_type == "application/json" and content_length is not None:
            try:
                too_large = int(content_length) > MAX_JSON_BODY_BYTES
            except ValueError:
                too_large = False
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content=error_payload("request_too_large", "request body exceeds 2 MB"),
                )
        return await call_next(request)

    if hasattr(application, "build_ingest_atom_handler"):
        ingest_handler = application.build_ingest_atom_handler()
    else:
        if projection_names is None:
            ingest_handler = build_ingest_atom_handler(database_path=app.state.database_path)
        else:
            ingest_handler = build_ingest_atom_handler(
                database_path=app.state.database_path,
                projection_names=projection_names,
            )
    if hasattr(application, "build_ingest_raw_round_handler"):
        raw_round_handler = application.build_ingest_raw_round_handler()
    else:
        raw_round_handler = build_ingest_raw_round_handler(
            database_path=app.state.database_path,
            max_payload_bytes=settings.max_raw_round_bytes,
            retention_days=settings.raw_round_retention_days,
        )
    if hasattr(application, "build_get_atom_handler"):
        get_atom_handler = application.build_get_atom_handler()
    else:
        get_atom_handler = build_get_atom_handler(database_path=app.state.database_path)
    if hasattr(application, "build_get_knowledge_handler"):
        get_knowledge_handler = application.build_get_knowledge_handler()
    else:
        get_knowledge_handler = build_get_knowledge_handler(
            database_path=app.state.database_path,
        )
    if hasattr(application, "build_get_knowledge_history_handler"):
        get_knowledge_history_handler = application.build_get_knowledge_history_handler()
    else:
        get_knowledge_history_handler = build_get_knowledge_history_handler(
            database_path=app.state.database_path,
        )
    if hasattr(application, "build_get_knowledge_evidence_handler"):
        get_knowledge_evidence_handler = application.build_get_knowledge_evidence_handler()
    else:
        get_knowledge_evidence_handler = build_get_knowledge_evidence_handler(
            database_path=app.state.database_path,
        )
    if hasattr(application, "build_retrieve_context_handler"):
        retrieve_context_handler = application.build_retrieve_context_handler()
    else:
        retrieve_context_handler = build_retrieve_context_handler(
            database_path=app.state.database_path,
        )
    app.state.ingest_atom_handler = ingest_handler
    app.state.raw_round_handler = raw_round_handler
    app.state.get_atom_handler = get_atom_handler
    app.state.get_knowledge_handler = get_knowledge_handler
    app.state.retrieve_context_handler = retrieve_context_handler

    require_token = build_bearer_token_dependency(settings=settings)
    maybe_token = build_optional_bearer_token_dependency(settings=settings)
    app.add_exception_handler(AuthenticationError, authentication_error_handler)  # type: ignore[arg-type]

    @app.get("/v1/ping")
    def ping(_token: str = Depends(require_token)) -> dict[str, str]:
        return {"pong": "true"}

    app.include_router(create_atoms_router(require_token, ingest_handler))
    app.include_router(create_rounds_router(require_token, raw_round_handler))
    app.include_router(
        create_knowledge_router(
            require_token,
            get_atom_handler=get_atom_handler,
            get_knowledge_handler=get_knowledge_handler,
            get_knowledge_history_handler=get_knowledge_history_handler,
            get_knowledge_evidence_handler=get_knowledge_evidence_handler,
        )
    )
    app.include_router(
        create_health_router(
            require_token,
            maybe_token,
            database_path=app.state.database_path,
            service_lock_path=settings.service_lock_path,
            write_handler=ingest_handler,
        )
    )
    app.include_router(
        create_context_router(
            require_token,
            retrieve_context_handler=retrieve_context_handler,
        )
    )

    return app
