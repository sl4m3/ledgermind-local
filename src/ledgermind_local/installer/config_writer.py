"""Persist installer config, profiles, and secret references atomically."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ledgermind_local.config import EmbeddingConfig as LocalEmbeddingConfig
from ledgermind_local.config import LocalConfig, ProfileSlotsConfig
from ledgermind_local.inference.profiles import ProviderKind

from .models import InstallerConfig
from .paths import InstallerPaths
from .permissions import ensure_private_dir
from .profiles.generation import build_generation_profiles
from .secret_refs import (
    EMBEDDING_SECRET_REF,
    GENERATION_SECRET_REF,
    LEGACY_EMBEDDING_SECRET_REF,
    LEGACY_GENERATION_SECRET_REF,
    LOCAL_EMBEDDING_SECRET_REF,
)
from .secrets.base import SecretBackend
from .secrets.file_store import FileSecretStore


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


def _ensure_local_server_token(paths: InstallerPaths) -> Path:
    """Create the bearer token consumed by the Local API and agent adapters."""

    target = paths.data_dir / "local" / "server.token"
    ensure_private_dir(target.parent)
    try:
        current = target.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        current = ""
    if not current:
        target.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def _write_local_embedding_token(paths: InstallerPaths, token: str) -> Path:
    target = paths.data_dir / "embedding" / "server.token"
    ensure_private_dir(target.parent)
    target.write_text(token + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def select_secret_backend(paths: InstallerPaths) -> SecretBackend:
    """Return the one canonical provider-secret store used by Local.

    Runtime inference always reads ``paths.secrets_file``.  Selecting Secret
    Service opportunistically here allowed the two stores to diverge and made
    an update copy a stale desktop-keyring value over the working runtime
    credential.  Keep one source of truth regardless of DBus availability.
    """

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


def _stored_secret_value(
    backend: SecretBackend, *secret_refs: str | None
) -> str | None:
    for secret_ref in secret_refs:
        if secret_ref:
            stored = backend.get(secret_ref)
            if stored:
                return stored
    return None


def resolve_provider_tokens(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
) -> tuple[str, str | None]:
    backend = select_secret_backend(paths)
    generation = (
        config.generation.token
        or (
            os.environ.get(config.generation.token_env, "")
            if config.generation.token_env
            else None
        )
        or (generation_stdin if config.generation.token_stdin else None)
        or _stored_secret_value(
            backend,
            config.generation.secret_ref,
            GENERATION_SECRET_REF,
            LEGACY_GENERATION_SECRET_REF,
        )
    )
    if not generation:
        raise ValueError("a provider token could not be resolved")
    embedding: str | None = None
    if config.embedding.mode == "api" and config.embedding.api is not None:
        embedding = (
            config.embedding.api.token
            or (
                os.environ.get(config.embedding.api.token_env, "")
                if config.embedding.api.token_env
                else None
            )
            or (embedding_stdin if config.embedding.api.token_stdin else None)
            or _stored_secret_value(
                backend,
                config.embedding.api.secret_ref,
                EMBEDDING_SECRET_REF,
                LEGACY_EMBEDDING_SECRET_REF,
            )
        )
        if not embedding:
            raise ValueError("a provider token could not be resolved")
    return generation, embedding


def _safe_config_payload(config: InstallerConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude_none=True)
    generation = dict(payload.get("generation", {}))
    generation.pop("token", None)
    generation.pop("token_env", None)
    generation.pop("token_stdin", None)
    generation["secret_ref"] = GENERATION_SECRET_REF
    payload["generation"] = generation
    embedding = dict(payload.get("embedding", {}))
    api = embedding.get("api")
    if isinstance(api, dict):
        api = dict(api)
        api.pop("token", None)
        api.pop("token_env", None)
        api.pop("token_stdin", None)
        api["secret_ref"] = EMBEDDING_SECRET_REF
        embedding["api"] = api
    payload["embedding"] = embedding
    return payload


def write_installer_config(
    config: InstallerConfig,
    paths: InstallerPaths,
    *,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
    preserve_provider_credentials: bool = False,
) -> dict[str, Any]:
    """Write secret-free config and profile bindings; return safe metadata."""

    paths.ensure()
    runtime_token = _ensure_runtime_token(paths)
    server_token = _ensure_local_server_token(paths)
    backend = select_secret_backend(paths)
    generation_ref = GENERATION_SECRET_REF
    if preserve_provider_credentials:
        if not backend.get(generation_ref):
            raise ValueError("the installed generation credential is missing")
    else:
        generation_token = _secret_value(
            config.generation.token,
            config.generation.token_env,
            generation_stdin if config.generation.token_stdin else None,
            secret_ref=config.generation.secret_ref,
            backend=backend,
        )
        backend.put(generation_ref, generation_token)
    profiles: list[dict[str, Any]] = []
    local_embedding_token_file: Path | None = None
    for generation_profile in build_generation_profiles(config.generation):
        profiles.append(
            {
                **generation_profile,
                "secret_ref": generation_ref,
            }
        )
    if config.embedding.mode == "api":
        assert config.embedding.api is not None
        embedding_ref = EMBEDDING_SECRET_REF
        if preserve_provider_credentials:
            if not backend.get(embedding_ref):
                raise ValueError("the installed embedding credential is missing")
        else:
            embedding_token = _secret_value(
                config.embedding.api.token,
                config.embedding.api.token_env,
                embedding_stdin if config.embedding.api.token_stdin else None,
                secret_ref=config.embedding.api.secret_ref,
                backend=backend,
            )
            backend.put(embedding_ref, embedding_token)
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
        local_embedding_token = backend.get(LOCAL_EMBEDDING_SECRET_REF)
        if not local_embedding_token:
            local_embedding_token = secrets.token_urlsafe(32)
            backend.put(LOCAL_EMBEDDING_SECRET_REF, local_embedding_token)
        local_embedding_token_file = _write_local_embedding_token(
            paths, local_embedding_token
        )
        profiles.append(
            {
                "profile_id": "embedding-local",
                "slot": "embedding",
                "endpoint": "http://127.0.0.1:8766",
                "model": config.embedding.local.catalog_id,
                "secret_ref": LOCAL_EMBEDDING_SECRET_REF,
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
        "secrets": str(paths.secrets_file),
        "runtime_token_file": str(runtime_token),
        "server_token_file": str(server_token),
        "local_embedding_token_file": (
            str(local_embedding_token_file)
            if local_embedding_token_file is not None
            else None
        ),
        "profile_ids": [item["profile_id"] for item in profiles],
    }


def persist_installer_config(config: InstallerConfig, paths: InstallerPaths) -> Path:
    """Persist an already-secret-free installer configuration.

    Integration lifecycle commands use this helper so enabling or disabling an
    adapter never has to resolve, receive, or rewrite provider credentials.
    """

    paths.ensure()
    _write_private_json(paths.config_file, _safe_config_payload(config))
    return paths.config_file


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
            provider_mode="api",
            endpoint="http://127.0.0.1:8766",
            model=config.embedding.local.catalog_id,
            dimensions=config.embedding.local.dimensions,
            batch_size=config.embedding.local.batch_size,
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
            secret_ref=EMBEDDING_SECRET_REF,
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
        if config.memory_mode == "shared":
            memory_spaces = {"shared": config.memory_space_id_for("hermes")}
        else:
            sources = {"hermes", *(item.id for item in config.integrations)}
            memory_spaces = {
                source: config.memory_space_id_for(source) for source in sources
            }
        repository = SQLiteMemorySpaceRepository(connection)
        for source, memory_space_id in memory_spaces.items():
            repository.ensure(memory_space_id, source)
        max_retries = config.advanced.retry_attempts
        max_input = config.advanced.generation_max_input or 12_000
        # Core's largest built-in semantic contract is the 2,048-token
        # execution pass. The installer default must be able to execute every
        # built-in contract without requiring an undocumented override.
        max_output = config.advanced.generation_max_output or 2_048
        profile_ids: list[str] = []
        for profile in build_generation_profiles(config.generation):
            materialized = InferenceProfile(
                profile_id=str(profile["profile_id"]),
                provider_kind=cast(ProviderKind, profile["provider_kind"]),
                base_url=str(profile["endpoint"]),
                model=str(profile["model"]),
                secret_ref=GENERATION_SECRET_REF,
                timeout_seconds=config.generation.timeout_seconds,
                max_retries=max_retries,
                max_input_tokens=max_input,
                max_output_tokens=max_output,
                extra_body=dict(profile.get("extra_body", {})),
                structured_output_preference="strict_json_schema",
            )
            store.upsert(materialized)
            for memory_space_id in memory_spaces.values():
                store.bind_slot(
                    memory_space_id,
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
                secret_ref=EMBEDDING_SECRET_REF,
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
                secret_ref=LOCAL_EMBEDDING_SECRET_REF,
                timeout_seconds=60.0,
                max_retries=max_retries,
                max_input_tokens=config.advanced.embedding_max_text_length or 12_000,
                max_output_tokens=2_000,
            )
        store.upsert(embedding_profile)
        for memory_space_id in memory_spaces.values():
            store.bind_slot(
                memory_space_id,
                slot="embedding",
                profile_id=embedding_profile.profile_id,
            )
        profile_ids.append(embedding_profile.profile_id)
        connection.commit()
    finally:
        connection.close()
    return {
        "database": str(database_path),
        "memory_space_id": config.memory_space_id_for("hermes"),
        "memory_space_ids": sorted(memory_spaces.values()),
        "profile_ids": profile_ids,
    }


def persist_generation_probe(
    config: InstallerConfig,
    paths: InstallerPaths,
    probe_result: dict[str, Any],
) -> dict[str, Any]:
    """Persist the installer's successful strict probe for runtime use.

    The installer and runtime share the exact materialized profile identity.
    This avoids both an unverified first semantic task and a second set of
    provider calls immediately after installation.
    """

    from ledgermind_local.inference.profile_store import InferenceProfileStore
    from ledgermind_local.inference.profiles import (
        ProviderCapabilities,
        generation_profile_fingerprint,
    )
    from ledgermind_local.persistence import open_sqlite_connection
    from ledgermind_local.persistence import rounds_migrations as migrations

    if probe_result.get("strict_structured_outputs") is not True:
        raise ValueError("generation probe did not verify strict structured outputs")
    probed_models = probe_result.get("probed_models")
    if not isinstance(probed_models, list) or not all(
        isinstance(item, str) and item for item in probed_models
    ):
        probed_models = [config.generation.model]
    verified_models = set(probed_models)
    local = build_local_config(config, paths)
    database_path = Path(local.rounds_database_path).expanduser()
    connection = open_sqlite_connection(database_path)
    persisted: list[str] = []
    try:
        migrations.apply_migrations(connection)
        store = InferenceProfileStore(connection)
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat(timespec="seconds")
        for profile_spec in build_generation_profiles(config.generation):
            profile_id = str(profile_spec["profile_id"])
            profile = store.get(profile_id)
            if profile is None:
                raise ValueError(
                    f"materialized generation profile is missing: {profile_id}"
                )
            if profile.model not in verified_models:
                raise ValueError(
                    f"generation profile model was not probed: {profile.model}"
                )
            store.upsert_capabilities(
                ProviderCapabilities(
                    profile_id=profile.profile_id,
                    profile_fingerprint=generation_profile_fingerprint(profile),
                    transport=profile.provider_kind,
                    model=profile.model,
                    structured_output_mode="strict_json_schema",
                    json_schema_supported=True,
                    structured_json_schema=True,
                    native_schema_strictness=True,
                    detected_capabilities={
                        "structured_json_schema": True,
                        "native_schema_strictness": True,
                    },
                    probe_contract_digest=str(
                        probe_result.get("probe_contract_digest", "")
                    )
                    or None,
                    probe_status="passed",
                    last_probed_at=timestamp,
                    probed_at=timestamp,
                    # Strict-schema support is bound to the complete,
                    # secret-free profile fingerprint.  It remains valid
                    # until that identity changes; expiring it by wall clock
                    # would disable an otherwise healthy unattended runtime.
                    expires_at=None,
                    probe_result="passed",
                )
            )
            persisted.append(profile.profile_id)
        connection.commit()
    finally:
        connection.close()
    return {"status": "passed", "profile_ids": persisted}


__all__ = [
    "build_local_config",
    "load_installer_config",
    "persist_generation_probe",
    "resolve_provider_tokens",
    "select_secret_backend",
    "write_installer_config",
    "write_local_config",
    "write_local_profiles",
]
