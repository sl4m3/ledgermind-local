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


class VectorProjectionConfig(BaseModel):
    """Configuration for optional local vector projection settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_path: str = "~/.ledgermind/local/models/v5-small-text-matching-Q4_K_M.gguf"
    gpu_layers: int = Field(default=0, ge=0)
    rebuild_threshold: int = Field(default=500, ge=0)


class MarkdownProjectionConfig(BaseModel):
    """Configuration for markdown derivative projection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class GitAuditConfig(BaseModel):
    """Configuration for markdown git-audit projection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class WorkerConfig(BaseModel):
    """Lifecycle settings shared by one guarded background worker."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interval_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)
    shutdown_timeout_seconds: float = Field(default=5.0, gt=0.0)


class WorkerSetConfig(BaseModel):
    """The complete set of Local-owned B2 worker switches."""

    model_config = ConfigDict(extra="forbid")

    retention: WorkerConfig = Field(
        default_factory=lambda: WorkerConfig(enabled=True, interval_seconds=300.0)
    )
    processing: WorkerConfig = Field(default_factory=WorkerConfig)
    core_commands: WorkerConfig = Field(
        default_factory=lambda: WorkerConfig(enabled=True)
    )
    core_projections: WorkerConfig = Field(
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


class SearchConfig(BaseModel):
    """Local candidate search policy before Core's authoritative ranking."""

    model_config = ConfigDict(extra="forbid")

    use_local_candidates: bool = True
    candidate_multiplier: int = Field(default=4, ge=1, le=100)
    fallback_to_core_fts: bool = True

    @property
    def enabled(self) -> bool:
        """Compatibility alias for the B2 ``use_local_candidates`` switch."""

        return self.use_local_candidates

    @property
    def fallback_to_core(self) -> bool:
        """Compatibility alias for the B2 Core FTS fallback switch."""

        return self.fallback_to_core_fts


class LocalConfig(BaseModel):
    """Strict service configuration for the local LedgerMind process."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(ge=1)
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
    search: SearchConfig = Field(default_factory=SearchConfig)
    log_level: str = "INFO"
    projection_poll_interval_seconds: float = Field(default=1.0, ge=0.0)
    processing_enabled: bool = False
    processing_poll_interval_seconds: float = Field(default=1.0, ge=0.0)
    processing_max_attempts: int = Field(default=3, ge=1)
    processing_retry_delay_seconds: float = Field(default=30.0, ge=0.0)
    processing_lease_seconds: float = Field(default=300.0, ge=1.0)
    processing_heartbeat_interval_seconds: float = Field(default=30.0, ge=1.0)
    hypothesis_profile_id: str | None = None
    inference_secrets_path: str = "secrets.json"
    max_raw_round_bytes: int = Field(default=5_000_000, ge=1)
    raw_round_retention_days: int = Field(default=30, ge=1)
    allow_remote_bind: bool = False
    vector: VectorProjectionConfig = Field(default_factory=VectorProjectionConfig)
    markdown_projection: MarkdownProjectionConfig = Field(
        default_factory=MarkdownProjectionConfig
    )
    markdown_audit: GitAuditConfig = Field(default_factory=GitAuditConfig)
    markdown_audit_enabled: bool = False

    @property
    def database_path(self) -> str:
        """Compatibility read alias; serialized config uses rounds name."""

        return self.rounds_database_path

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

    @field_validator("hypothesis_profile_id")
    @classmethod
    def _normalize_hypothesis_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("hypothesis_profile_id must not be empty when configured")
        return normalized

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
        # pylint: disable=unused-argument
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

        # A persisted configuration carrying the removed switch is an older
        # schema.  Always emit the current version after migration.
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
            # A legacy true value is an explicit security requirement.  If a
            # hand-edited config also supplied a permissive profile, migrate
            # to the fail-closed equivalent rather than weakening it.
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
    def _materialize_legacy_worker_settings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        workers_value = data.get("workers")
        if isinstance(workers_value, WorkerSetConfig):
            workers = workers_value.model_dump(mode="python")
        elif isinstance(workers_value, dict):
            workers = dict(workers_value)
        else:
            workers = {}
        if "processing" not in workers:
            workers["processing"] = {
                "enabled": bool(data.get("processing_enabled", False)),
                "interval_seconds": data.get("processing_poll_interval_seconds", 1.0),
            }
        if (
            bool(data.get("processing_enabled", False))
            and "core_commands" not in workers
        ):
            workers["core_commands"] = {"enabled": True}
        data["workers"] = workers
        return data

    @model_validator(mode="after")
    def _sync_legacy_worker_fields(self) -> LocalConfig:
        processing = self.workers.processing
        object.__setattr__(self, "processing_enabled", processing.enabled)
        object.__setattr__(
            self,
            "processing_poll_interval_seconds",
            processing.interval_seconds,
        )
        return self

    @model_validator(mode="after")
    def _sync_markdown_audit_flag(self) -> LocalConfig:
        if (
            self.markdown_projection.enabled
            or self.markdown_audit_enabled
            or self.markdown_audit.enabled
        ):
            object.__setattr__(self.markdown_projection, "enabled", True)
            object.__setattr__(self, "markdown_audit_enabled", True)
        else:
            object.__setattr__(
                self,
                "markdown_audit_enabled",
                bool(self.markdown_audit_enabled),
            )
        return self

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
            self.model_dump(
                exclude={"markdown_audit_enabled"},
                exclude_defaults=False,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
