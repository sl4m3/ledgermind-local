"""Bootstrap utilities for the Local service and its Rust Core boundary."""

from __future__ import annotations

import os
import secrets
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ledgermind_local.config import LocalConfig
from ledgermind_local.inference import InferenceBroker, SecretStore
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.processing import (
    BrokerHypothesisGenerator,
    HypothesisGenerator,
    RoundProcessingWorker,
)
from ledgermind_local.raw_rounds import RawRoundIngestHandler

DEFAULT_PROJECTIONS: tuple[str, ...] = (
    "projections.search",
    "projections.knowledge",
    "projections.markdown",
)


def build_projection_names(config: LocalConfig) -> tuple[str, ...]:
    names = list(DEFAULT_PROJECTIONS)
    if config.markdown_projection.enabled or config.markdown_audit_enabled:
        names.append("projections.markdown_audit")
    return tuple(names)


def build_core_projection_handlers(
    *,
    connection: sqlite3.Connection,
    database_path: str | Path,
    config: LocalConfig,
) -> dict[str, Any]:
    """Build Local-only handlers for public Rust Core projection events."""

    from ledgermind_local.projections import (
        KnowledgeFTSProjection,
        KnowledgeMarkdownProjection,
        KnowledgeVectorProjection,
    )

    handlers: dict[str, Any] = {
        "projections.search": KnowledgeFTSProjection(connection),
    }
    if config.vector.enabled:
        model_path = Path(config.vector.model_path).expanduser()
        from ledgermind_local.projections import GGUFVectorizer

        handlers["projections.knowledge"] = KnowledgeVectorProjection(
            connection=connection,
            vector_store_root=_build_vector_store_root(database_path),
            vectorizer_factory=lambda: GGUFVectorizer(
                model_path=model_path,
                gpu_layers=config.vector.gpu_layers,
            ),
        )
    if config.markdown_projection.enabled:
        handlers["projections.markdown"] = KnowledgeMarkdownProjection(
            connection=connection,
            markdown_root=_build_markdown_root(database_path),
        )
    return handlers


def _build_vector_store_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".vectors")


def _build_markdown_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".markdown")


def _build_vectorizer_factory() -> Callable[[], Any] | None:
    model_path = os.environ.get("LEDGERMIND_VECTOR_MODEL_PATH")
    if not model_path:
        return None

    from ledgermind_local.projections import GGUFVectorizer

    return lambda: GGUFVectorizer(model_path=model_path)


def build_ingest_raw_round_handler(
    *,
    database_path: str | Path,
    max_raw_round_bytes: int = 5_000_000,
    retention_days: int = 30,
    pipeline_version: int = 1,
    normalizer_version: int = 1,
    prompt_version: int = 1,
) -> RawRoundIngestHandler:
    """Build the Local-owned RawRound capture handler."""

    return RawRoundIngestHandler(
        database_path=database_path,
        max_raw_round_bytes=max_raw_round_bytes,
        retention_days=retention_days,
        pipeline_version=pipeline_version,
        normalizer_version=normalizer_version,
        prompt_version=prompt_version,
    )


def build_round_processing_worker(
    *,
    database_path: str | Path,
    generator: HypothesisGenerator | None = None,
    broker: InferenceBroker | None = None,
    hypothesis_profile_id: str | None = None,
    secrets_path: str | Path | None = None,
    worker_id: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 30,
    lease_seconds: float = 300,
    heartbeat_interval_seconds: float = 30,
) -> RoundProcessingWorker:
    """Build Local hypothesis generation and durable Core-command worker."""

    if generator is not None:
        selected_generator = generator
    else:
        if not hypothesis_profile_id:
            raise ValueError(
                "hypothesis_profile_id is required when processing uses the inference broker"
            )
        selected_broker = broker
        if selected_broker is None:
            resolved_secrets_path = (
                Path(secrets_path)
                if secrets_path is not None
                else Path(database_path).parent / "secrets.json"
            )
            selected_broker = InferenceBroker(
                database_path=database_path,
                secret_store=SecretStore(resolved_secrets_path),
            )
        profile = selected_broker.get_profile(hypothesis_profile_id)
        selected_generator = BrokerHypothesisGenerator(
            selected_broker,
            profile_id=hypothesis_profile_id,
            provider=profile.provider_kind,
            model=profile.model,
            prompt_version=profile.hypothesis_prompt_version,
            schema_version=profile.hypothesis_schema_version,
        )
    return RoundProcessingWorker(
        database_path=database_path,
        generator=selected_generator,
        worker_id=worker_id,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def bootstrap_local_service(
    *,
    home: str | Path = "~/.ledgermind/local",
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig]:
    """Prepare Local filesystem and return resolved runtime objects."""

    paths = ServicePaths(home=home)
    cfg = config or LocalConfig(config_version=1)
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    database_path = paths.resolve_rounds_database_path(cfg.rounds_database_path)
    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(database_path)
    connection.close()
    return paths, cfg


def _atomic_write_text(path: Path, data: str, *, mode: int) -> None:
    """Write text atomically with explicit permissions."""

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
            os.chmod(tmp_path, mode)
            tmp_path.replace(path)
            os.chmod(path, mode)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def initialize_local_layout(
    *,
    home: str | Path = "~/.ledgermind/local",
    force: bool = False,
    rotate_token: bool = False,
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig, str]:
    """Initialize Local directories, config and API token."""

    preloaded_config = config
    if preloaded_config is None and not force:
        candidate = ServicePaths(home=home).config_file
        if candidate.exists():
            preloaded_config = LocalConfig.from_file(candidate)
    paths, cfg = bootstrap_local_service(home=home, config=preloaded_config)

    config_path = paths.config_file
    if force or not config_path.exists():
        cfg = config or LocalConfig(config_version=1)
        _atomic_write_text(config_path, cfg.to_json(), mode=0o600)
    elif config is None:
        cfg = LocalConfig.from_file(config_path)
    else:
        cfg = config

    database_path = paths.resolve_rounds_database_path(cfg.rounds_database_path)
    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(database_path)
    connection.close()

    if not paths.token_file.exists() or (force and rotate_token):
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    if rotate_token:
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    existing_token = paths.token_file.read_text(encoding="utf-8")
    return paths, cfg, existing_token