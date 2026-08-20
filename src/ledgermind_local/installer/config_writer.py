"""Persist installer config, profiles, and secret references atomically."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from ledgermind_local.config import EmbeddingConfig as LocalEmbeddingConfig
from ledgermind_local.config import LocalConfig, ProfileSlotsConfig

from .models import InstallerConfig
from .paths import InstallerPaths
from .permissions import ensure_private_dir
from .profiles.generation import build_generation_profiles
from .secrets.base import SecretBackend
from .secrets.file_store import FileSecretStore
from .secrets.secret_service import SecretServiceStore


def _write_private_json(path: Path, payload: object) -> None:
    ensure_private_dir(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_runtime_token(paths: InstallerPaths) -> Path:
    target = paths.config_dir / "runtime.token"
    if not target.exists():
        ensure_private_dir(target.parent)
        target.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def select_secret_backend(paths: InstallerPaths) -> SecretBackend:
    if os.environ.get("LEDGERMIND_SECRET_BACKEND", "").lower() == "file":
        return FileSecretStore(paths.secrets_file)
    service = SecretServiceStore()
    if service.available():
        return service
    return FileSecretStore(paths.secrets_file)


def _secret_value(
    value: str | None,
    env_name: str | None,
    stdin_value: str | None,
    *,
    secret_ref: str | None,
    backend: SecretBackend,
) -> str:
    if value:
        return value
    if env_name:
        resolved = os.environ.get(env_name, "")
        if resolved:
            return resolved
    if stdin_value:
        return stdin_value
    if secret_ref:
        stored = backend.get(secret_ref)
        if stored:
            return stored
    raise ValueError("a provider token could not be resolved")


def resolve_provider_tokens(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
) -> tuple[str, str | None]:
    backend = select_secret_backend(paths)
    generation = _secret_value(
        config.generation.token,
        config.generation.token_env,
        generation_stdin if config.generation.token_stdin else None,
        secret_ref=config.generation.secret_ref or "generation/token",
        backend=backend,
    )
    embedding: str | None = None
    if config.embedding.mode == "api" and config.embedding.api is not None:
        embedding = _secret_value(
            config.embedding.api.token,
            config.embedding.api.token_env,
            embedding_stdin if config.embedding.api.token_stdin else None,
            secret_ref=config.embedding.api.secret_ref or "embedding/token",
            backend=backend,
        )
    return generation, embedding


def _safe_config_payload(config: InstallerConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude_none=True)
    generation = dict(payload.get("generation", {}))
    generation.pop("token", None)
    generation.pop("token_env", None)
    generation.pop("token_stdin", None)
    generation["secret_ref"] = "generation/token"
    payload["generation"] = generation
    embedding = dict(payload.get("embedding", {}))
    api = embedding.get("api")
    if isinstance(api, dict):
        api = dict(api)
        api.pop("token", None)
        api.pop("token_env", None)
        api.pop("token_stdin", None)
        api["secret_ref"] = "embedding/token"
        embedding["api"] = api
    payload["embedding"] = embedding
    return payload


def write_installer_config(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
) -> dict[str, Any]:
    """Write secret-free config and profile bindings; return safe metadata."""

    paths.ensure()
    runtime_token = _ensure_runtime_token(paths)
    backend = select_secret_backend(paths)
    generation_token = _secret_value(
        config.generation.token,
        config.generation.token_env,
        generation_stdin if config.generation.token_stdin else None,
        secret_ref=config.generation.secret_ref,
        backend=backend,
    )
    generation_ref = "generation/token"
    backend.put(generation_ref, generation_token)
    if isinstance(backend, SecretServiceStore):
        FileSecretStore(paths.secrets_file).put(generation_ref, generation_token)
    profiles: list[dict[str, Any]] = []
    for generation_profile in build_generation_profiles(config.generation):
        profiles.append(
            {
                **generation_profile,
                "secret_ref": generation_ref,
            }
        )
    if config.embedding.mode == "api":
        assert config.embedding.api is not None
        embedding_token = _secret_value(
            config.embedding.api.token,
            config.embedding.api.token_env,
            embedding_stdin if config.embedding.api.token_stdin else None,
            secret_ref=config.embedding.api.secret_ref,
            backend=backend,
        )
        embedding_ref = "embedding/token"
        backend.put(embedding_ref, embedding_token)
        if isinstance(backend, SecretServiceStore):
            FileSecretStore(paths.secrets_file).put(embedding_ref, embedding_token)
        profiles.append(
            {
                "profile_id": "embedding-default",
                "slot": "embedding",
                "endpoint": config.embedding.api.endpoint,
                "model": config.embedding.api.model,
                "secret_ref": embedding_ref,
                "dimensions": config.embedding.api.dimensions,
                "batch_size": config.embedding.api.batch_size,
                "concurrency": config.embedding.api.concurrency,
            }
        )
    else:
        assert config.embedding.local is not None
        profiles.append(
            {
                "profile_id": "embedding-local",
                "slot": "embedding",
                "endpoint": "http://127.0.0.1:8766",
                "model": config.embedding.local.catalog_id,
                "secret_ref": None,
                "device": config.embedding.local.device,
                "batch_size": config.embedding.local.batch_size,
                "concurrency": config.embedding.local.concurrency,
            }
        )
    _write_private_json(paths.config_file, _safe_config_payload(config))
    _write_private_json(paths.profiles_file, {"profiles": profiles})
    return {
        "config": str(paths.config_file),
        "profiles": str(paths.profiles_file),
        "service_config": str(paths.data_dir / "local" / "config.json"),
        "secrets": str(paths.secrets_file)
        if isinstance(backend, FileSecretStore)
        else "secret-service",
        "runtime_token_file": str(runtime_token),
        "profile_ids": [item["profile_id"] for item in profiles],
    }


def load_installer_config(path: str | Path) -> InstallerConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return InstallerConfig.model_validate(payload)


def build_local_config(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    release_dir: str | Path | None = None,
) -> LocalConfig:
    """Translate installer settings into the existing Local runtime contract."""

    release = (
        Path(release_dir).expanduser()
        if release_dir is not None
        else paths.current_link
    )
    embedding = LocalEmbeddingConfig(enabled=False)
    if config.embedding.mode == "local" and config.embedding.local is not None:
        model_path = config.embedding.local.model_storage_path or str(
            paths.models_dir / config.embedding.local.catalog_id
        )
        device = config.embedding.local.device
        if device == "auto":
            device = "cpu"
        embedding = LocalEmbeddingConfig(
            enabled=True,
            model_path=model_path,
            gpu_layers=99 if device != "cpu" else 0,
        )
    elif config.embedding.mode == "api" and config.embedding.api is not None:
        embedding = LocalEmbeddingConfig(
            enabled=True,
            provider_mode="api",
            endpoint=config.embedding.api.endpoint,
            model=config.embedding.api.model,
            dimensions=config.embedding.api.dimensions,
            batch_size=config.embedding.api.batch_size,
            timeout_seconds=config.embedding.api.timeout_seconds,
            secret_ref="embedding/token",
        )
    memory_root = (
        Path(config.memory_data_path).expanduser()
        if config.memory_data_path is not None
        else paths.memory_data_dir
    )
    return LocalConfig(
        config_version=2,
        semantic_language=config.semantic_language,
        rounds_database_path=str(memory_root / "rounds.db"),
        knowledge_database_path=str(memory_root / "core" / "knowledge.db"),
        core_binary_path=str(release / "bin" / "ledgermind-core"),
        core_signature_path=str(release / "signatures" / "ledgermind-core.sig"),
        core_public_key_path=str(release / "signatures" / "ledgermind-core.pub"),
        inference_secrets_path=str(paths.secrets_file),
        profile_slots=ProfileSlotsConfig(
            operational="generation-operational",
            object_resolution="generation-object-resolution",
            background="generation-background",
            embedding=(
                "embedding-default"
                if config.embedding.mode == "api"
                else "embedding-local"
            ),
        ),
        embedding=embedding,
    )


def write_local_config(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    release_dir: str | Path | None = None,
) -> Path:
    target = paths.config_dir / "local-config.json"
    local = build_local_config(config, paths, release_dir=release_dir)
    payload = json.loads(local.to_json())
    _write_private_json(target, payload)
    _write_private_json(paths.data_dir / "local" / "config.json", payload)
    return target


def write_local_profiles(
    config: InstallerConfig,
    paths: InstallerPaths,
) -> dict[str, Any]:
    """Materialize installer profiles in Local's existing SQLite resolver tables."""

    from ledgermind_local.inference.profile_store import InferenceProfileStore
    from ledgermind_local.inference.profiles import InferenceProfile
    from ledgermind_local.persistence import open_sqlite_connection
    from ledgermind_local.persistence import rounds_migrations as migrations
    from ledgermind_local.persistence.memory_space_repository import (
        SQLiteMemorySpaceRepository,
    )

    local = build_local_config(config, paths)
    database_path = Path(local.rounds_database_path).expanduser()
    ensure_private_dir(database_path.parent)
    connection = open_sqlite_connection(database_path)
    try:
        migrations.apply_migrations(connection)
        store = InferenceProfileStore(connection)
        SQLiteMemorySpaceRepository(connection).ensure("hermes-default", "hermes")
        max_retries = config.advanced.retry_attempts
        max_input = config.advanced.generation_max_input or 12_000
        max_output = config.advanced.generation_max_output or 2_000
        profile_ids: list[str] = []
        for profile in build_generation_profiles(config.generation):
            materialized = InferenceProfile(
                profile_id=str(profile["profile_id"]),
                base_url=str(profile["endpoint"]),
                model=str(profile["model"]),
                secret_ref="generation/token",
                timeout_seconds=config.generation.timeout_seconds,
                max_retries=max_retries,
                max_input_tokens=max_input,
                max_output_tokens=max_output,
            )
            store.upsert(materialized)
            store.bind_slot(
                "hermes-default",
                slot=str(profile["slot"]),
                profile_id=materialized.profile_id,
            )
            profile_ids.append(materialized.profile_id)

        if config.embedding.mode == "api" and config.embedding.api is not None:
            embedding = config.embedding.api
            embedding_profile = InferenceProfile(
                profile_id="embedding-default",
                base_url=embedding.endpoint,
                model=embedding.model,
                secret_ref="embedding/token",
                timeout_seconds=embedding.timeout_seconds,
                max_retries=max_retries,
                max_input_tokens=config.advanced.embedding_max_text_length or 12_000,
                max_output_tokens=2_000,
            )
        else:
            assert config.embedding.local is not None
            embedding_profile = InferenceProfile(
                profile_id="embedding-local",
                base_url="http://127.0.0.1:8766",
                model=config.embedding.local.catalog_id,
                secret_ref="embedding/local",
                timeout_seconds=60.0,
                max_retries=max_retries,
                max_input_tokens=config.advanced.embedding_max_text_length or 12_000,
                max_output_tokens=2_000,
            )
        store.upsert(embedding_profile)
        store.bind_slot(
            "hermes-default",
            slot="embedding",
            profile_id=embedding_profile.profile_id,
        )
        profile_ids.append(embedding_profile.profile_id)
        connection.commit()
    finally:
        connection.close()
    return {
        "database": str(database_path),
        "memory_space_id": "hermes-default",
        "profile_ids": profile_ids,
    }


__all__ = [
    "build_local_config",
    "load_installer_config",
    "resolve_provider_tokens",
    "select_secret_backend",
    "write_installer_config",
    "write_local_config",
    "write_local_profiles",
]
