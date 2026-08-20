"""Local service configuration models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

CURRENT_CONFIG_VERSION = 2


class EmbeddingConfig(BaseModel):
    """Configuration for the Local technical embedding backend."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_path: str = "~/.ledgermind/local/models/small-text-matching-Q4_K_M.gguf"
    gpu_layers: int = Field(default=0, ge=0)
    provider_mode: Literal["local", "api"] = "local"
    endpoint: str | None = None
    model: str | None = None
    dimensions: int = Field(default=0, ge=0)
    batch_size: int = Field(default=32, ge=1, le=4096)
    max_concurrency: int = Field(default=1, ge=1, le=32)
    max_wait_seconds: float = Field(default=0.05, ge=0.0, le=60.0)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    secret_ref: str | None = None


class WorkerConfig(BaseModel):
    """Lifecycle settings shared by one guarded background worker."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interval_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)
    shutdown_timeout_seconds: float = Field(default=5.0, gt=0.0)


class WorkerSetConfig(BaseModel):
    """The complete set of current Local-owned worker switches."""

    model_config = ConfigDict(extra="forbid")

    retention: WorkerConfig = Field(
        default_factory=lambda: WorkerConfig(enabled=True, interval_seconds=300.0)
    )
    core_commands: WorkerConfig = Field(
        default_factory=lambda: WorkerConfig(enabled=True)
    )
    core_model_tasks: WorkerConfig = Field(
        default_factory=lambda: WorkerConfig(enabled=True)
    )


class CoreSecurityConfig(BaseModel):
    """Explicit Core launch security profile and required guarantees."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    profile: Literal["secure", "permissive"] = "secure"
    require_network_isolation: bool = False
    require_rounds_database_hidden: bool = False
    require_filesystem_allowlist: bool = False
    require_environment_sanitized: bool = False
    require_signature: bool = False

    @model_validator(mode="before")
    @classmethod
    def _materialize_profile_defaults(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        profile = data.get("profile", "secure")
        secure_defaults = {
            "require_network_isolation": True,
            "require_rounds_database_hidden": True,
            "require_filesystem_allowlist": True,
            "require_environment_sanitized": True,
            "require_signature": True,
        }
        if profile == "secure":
            for key, default in secure_defaults.items():
                if key in data and data[key] is False:
                    raise ValueError(f"secure profile requires {key}=true")
                data.setdefault(key, default)
        return data

    @model_validator(mode="after")
    def _validate_secure_guarantees(self) -> CoreSecurityConfig:
        if self.profile == "secure":
            for field_name in (
                "require_network_isolation",
                "require_rounds_database_hidden",
                "require_filesystem_allowlist",
                "require_environment_sanitized",
                "require_signature",
            ):
                if not getattr(self, field_name):
                    raise ValueError(f"secure profile requires {field_name}=true")
        return self


def _default_core_security() -> CoreSecurityConfig:
    return CoreSecurityConfig(
        profile="secure",
        require_network_isolation=True,
        require_rounds_database_hidden=True,
        require_filesystem_allowlist=True,
        require_environment_sanitized=True,
        require_signature=True,
    )


class ProfileSlotsConfig(BaseModel):
    """Optional persisted defaults for the four technical profile slots.

    Bindings are resolved from Local's profile store at task execution time.
    A missing ``object_resolution`` binding is therefore an explicit
    ``profile_missing`` error; this model never derives it from ``operational``.
    """

    model_config = ConfigDict(extra="forbid")

    operational: str | None = None
    object_resolution: str | None = None
    background: str | None = None
    embedding: str | None = None

    @field_validator("operational", "object_resolution", "background", "embedding")
    @classmethod
    def _normalize_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("profile slot IDs must not be empty")
        return normalized


class LocalConfig(BaseModel):
    """Strict service configuration for the current Local runtime."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(ge=1)
    # Runtime language is deployment configuration, not request metadata.
    semantic_language: Literal["ru", "en", "es", "pt", "fr", "de", "uk"]
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8765, ge=1, le=65535)
    rounds_database_path: str = Field(
        default="rounds.db",
        validation_alias=AliasChoices("rounds_database_path", "database_path"),
    )
    knowledge_database_path: str = "../core/knowledge.db"
    core_backend: Literal["process"] = "process"
    core_binary_path: str = "../core/bin/ledgermind-core"
    core_signature_path: str = "../core/bin/ledgermind-core.sig"
    core_public_key_path: str = "../core/bin/ledgermind-core.pub"
    verify_core_signature: bool = True
    core_request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    core_startup_timeout_seconds: float = Field(default=10.0, gt=0.0)
    core_security: CoreSecurityConfig = Field(default_factory=_default_core_security)
    workers: WorkerSetConfig = Field(default_factory=WorkerSetConfig)
    profile_slots: ProfileSlotsConfig = Field(default_factory=ProfileSlotsConfig)
    log_level: str = "INFO"
    worker_max_attempts: int = Field(default=3, ge=1)
    worker_retry_delay_seconds: float = Field(default=30.0, ge=0.0)
    worker_lease_seconds: float = Field(default=300.0, ge=1.0)
    generation_concurrency: int = Field(default=1, ge=1, le=32)
    provider_capability_ttl_seconds: int = Field(default=86_400, ge=1, le=31_536_000)
    inference_secrets_path: str = "secrets.json"
    max_raw_round_bytes: int = Field(default=5_000_000, ge=1)
    raw_round_retention_days: int = Field(default=30, ge=1)
    allow_remote_bind: bool = False
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @property
    def database_path(self) -> str:
        """Read alias retained for callers that use the Local database name."""

        return self.rounds_database_path

    @property
    def embedding_batch_size(self) -> int:
        """Portable name for the Local embedding batch-size setting."""

        return self.embedding.batch_size

    @property
    def embedding_batch_max_wait_ms(self) -> int:
        """Portable millisecond view of the embedding batch wait setting."""

        return round(self.embedding.max_wait_seconds * 1000)

    @property
    def embedding_concurrency(self) -> int:
        """Portable name for the Local embedding concurrency setting."""

        return self.embedding.max_concurrency

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("log_level must not be empty")
        return normalized

    @field_validator("bind_host")
    @classmethod
    def _validate_bind_host(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("bind_host must not be empty")
        return stripped

    @field_validator("inference_secrets_path")
    @classmethod
    def _normalize_inference_secrets_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("inference_secrets_path must not be empty")
        return normalized

    @field_validator("allow_remote_bind", mode="after")
    @classmethod
    def _validate_remote_bind(cls, value: bool, info: Any) -> bool:
        if not value and info.data.get("bind_host") not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "bind_host outside localhost requires allow_remote_bind=true"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_core_security(cls, value: object) -> object:
        """Consume the removed isolation switch without weakening old configs."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy = data.pop("require_core_network_isolation", None)
        if legacy is None:
            return data

        version = data.get("config_version")
        if not isinstance(version, int) or version < CURRENT_CONFIG_VERSION:
            data["config_version"] = CURRENT_CONFIG_VERSION

        security = data.get("core_security")
        if security is None:
            security = {
                "profile": "secure",
                "require_network_isolation": True,
                "require_rounds_database_hidden": True,
                "require_filesystem_allowlist": True,
                "require_environment_sanitized": True,
                "require_signature": True,
            }
        elif bool(legacy):
            if isinstance(security, CoreSecurityConfig):
                security = security.model_dump(mode="python")
            elif isinstance(security, dict):
                security = dict(security)
            else:
                security = {}
            security.update(
                {
                    "profile": "secure",
                    "require_network_isolation": True,
                    "require_rounds_database_hidden": True,
                    "require_filesystem_allowlist": True,
                    "require_environment_sanitized": True,
                    "require_signature": True,
                }
            )
        data["core_security"] = security
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_profile_slots(cls, value: object) -> object:
        """Map legacy profile names to technical slots exactly once."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_operational = data.pop("hypothesis_profile_id", None)
        legacy_background = data.pop("merge_profile_id", None)
        if legacy_operational is None and legacy_background is None:
            return data
        raw_slots = data.get("profile_slots")
        if isinstance(raw_slots, ProfileSlotsConfig):
            slots = raw_slots.model_dump(mode="python")
        elif isinstance(raw_slots, dict):
            slots = dict(raw_slots)
        else:
            slots = {}
        if legacy_operational is not None:
            slots.setdefault("operational", legacy_operational)
        if legacy_background is not None:
            slots.setdefault("background", legacy_background)
        data["profile_slots"] = slots
        version = data.get("config_version")
        if not isinstance(version, int) or version < CURRENT_CONFIG_VERSION:
            data["config_version"] = CURRENT_CONFIG_VERSION
        return data

    @model_validator(mode="before")
    @classmethod
    def _remove_legacy_runtime_settings(cls, value: object) -> object:
        """Drop removed workers/projections while preserving safe settings.

        Existing Local configs are accepted once, but removed switches are never
        represented on the current model and therefore cannot start old workers.
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)
        changed = False

        workers_value = data.get("workers")
        if isinstance(workers_value, WorkerSetConfig):
            workers = workers_value.model_dump(mode="python")
        elif isinstance(workers_value, dict):
            workers = dict(workers_value)
        else:
            workers = {}
        for name in ("processing", "core_projections"):
            if name in workers:
                workers.pop(name)
                changed = True
        if workers_value is not None:
            data["workers"] = workers

        scalar_migrations = {
            "processing_max_attempts": "worker_max_attempts",
            "processing_retry_delay_seconds": "worker_retry_delay_seconds",
            "processing_lease_seconds": "worker_lease_seconds",
        }
        for old_name, new_name in scalar_migrations.items():
            if old_name in data:
                if new_name not in data:
                    data[new_name] = data[old_name]
                data.pop(old_name)
                changed = True

        for old_name in (
            "processing_enabled",
            "processing_poll_interval_seconds",
            "processing_heartbeat_interval_seconds",
            "projection_poll_interval_seconds",
            "search",
            "markdown_projection",
            "markdown_audit",
            "markdown_audit_enabled",
        ):
            if old_name in data:
                data.pop(old_name)
                changed = True

        legacy_embedding = data.pop("vector", None)
        if legacy_embedding is not None:
            changed = True
            if "embedding" not in data and isinstance(legacy_embedding, dict):
                data["embedding"] = {
                    key: legacy_embedding[key]
                    for key in ("enabled", "model_path", "gpu_layers")
                    if key in legacy_embedding
                }

        if changed:
            version = data.get("config_version")
            if not isinstance(version, int) or version < CURRENT_CONFIG_VERSION:
                data["config_version"] = CURRENT_CONFIG_VERSION
        return data

    @classmethod
    def from_json(cls, payload: str) -> LocalConfig:
        config = cls.model_validate_json(payload)
        return cls._upgrade_persisted_version(config)

    @classmethod
    def from_file(cls, path: Path) -> LocalConfig:
        config = cls.model_validate_json(path.read_text(encoding="utf-8"))
        return cls._upgrade_persisted_version(config)

    @classmethod
    def from_dict(cls, payload: object) -> LocalConfig:
        config = cls.model_validate(payload)
        return cls._upgrade_persisted_version(config)

    @staticmethod
    def _upgrade_persisted_version(config: LocalConfig) -> LocalConfig:
        if config.config_version < CURRENT_CONFIG_VERSION:
            return config.model_copy(update={"config_version": CURRENT_CONFIG_VERSION})
        return config

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(exclude_defaults=False),
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )
