"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from bootstrap import (
    build_get_atom_handler,
    build_get_knowledge_handler,
    build_ingest_atom_handler,
    build_get_knowledge_history_handler,
    build_get_knowledge_evidence_handler,
    build_retrieve_context_handler,
)

from .auth import (
    build_bearer_token_dependency,
    build_optional_bearer_token_dependency,
)
from .atoms import create_atoms_router
from .health import create_health_router
from .context import create_context_router
from .knowledge import create_knowledge_router
from .dependencies import Application, Settings
from .errors import AuthenticationError, authentication_error_handler


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
    app.state.get_atom_handler = get_atom_handler
    app.state.get_knowledge_handler = get_knowledge_handler
    app.state.retrieve_context_handler = retrieve_context_handler

    require_token = build_bearer_token_dependency(settings=settings)
    maybe_token = build_optional_bearer_token_dependency(settings=settings)
    app.add_exception_handler(AuthenticationError, authentication_error_handler)

    @app.get("/v1/ping")
    def ping(_token: str = Depends(require_token)) -> dict[str, str]:
        return {"pong": "true"}

    app.include_router(create_atoms_router(require_token, ingest_handler))
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
