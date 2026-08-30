"""Typed installer configuration and result-facing models."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _http_url(value: str, field_name: str = "endpoint") -> str:
    normalized = _text(value, field_name).rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http(s) URL")
    return normalized


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    token: str | None = None
    token_env: str | None = None
    token_stdin: bool = False
    secret_ref: str | None = None
    model: str
    operational_model: str | None = None
    # Object Resolution is a separate production role.  It is optional only
    # at parse time so legacy installer payloads can be diagnosed explicitly;
    # profile materialization refuses to invent an operational fallback.
    object_resolution_model: str | None = None
    background_model: str | None = None
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    structured_json_support: bool = True

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _http_url(value)

    @field_validator("model", "operational_model", "object_resolution_model", "background_model")
    @classmethod
    def validate_models(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _text(value, getattr(info, "field_name", "model"))

    @model_validator(mode="after")
    def validate_secret_source(self) -> GenerationConfig:
        if not (self.token or self.token_env or self.token_stdin or self.secret_ref):
            raise ValueError(
                "generation requires token, token_env, token_stdin, or secret_ref"
            )
        if self.token and self.token_env:
            raise ValueError("generation token and token_env are mutually exclusive")
        return self


class EmbeddingApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    token: str | None = None
    token_env: str | None = None
    token_stdin: bool = False
    secret_ref: str | None = None
    model: str
    dimensions: int = Field(gt=0)
    batch_size: int = Field(default=32, ge=1, le=4096)
    concurrency: int = Field(default=1, ge=1, le=64)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _http_url(value)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _text(value, "model")

    @model_validator(mode="after")
    def validate_secret_source(self) -> EmbeddingApiConfig:
        if not (self.token or self.token_env or self.token_stdin or self.secret_ref):
            raise ValueError(
                "embedding API requires token, token_env, token_stdin, or secret_ref"
            )
        if self.token and self.token_env:
            raise ValueError("embedding token and token_env are mutually exclusive")
        return self


class LocalEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_id: str
    device: Literal["auto", "cpu", "cuda", "rocm"] = "auto"
    model_storage_path: str | None = None
    model_path: str | None = None
    batch_size: int = Field(default=32, ge=1, le=4096)
    concurrency: int = Field(default=1, ge=1, le=64)
    threads: int = Field(default=0, ge=0, le=512)
    gpu_allocation: str = "balanced"
    auto_start: bool = True

    @field_validator("catalog_id", "gpu_allocation")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _text(value, getattr(info, "field_name", "value"))


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["api", "local"]
    api: EmbeddingApiConfig | None = None
    local: LocalEmbeddingConfig | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> EmbeddingConfig:
        if self.mode == "api" and self.api is None:
            raise ValueError("embedding.api is required in API mode")
        if self.mode == "local" and self.local is None:
            raise ValueError("embedding.local is required in local mode")
        if self.mode == "api" and self.local is not None:
            raise ValueError("embedding.local is not allowed in API mode")
        if self.mode == "local" and self.api is not None:
            raise ValueError("embedding.api is not allowed in local mode")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_shutdown_seconds: float = Field(default=60.0, ge=0, le=86_400)
    lease_ttl_seconds: float = Field(default=30.0, gt=0, le=3_600)
    heartbeat_seconds: float = Field(default=10.0, gt=0, le=1_200)

    @model_validator(mode="after")
    def validate_heartbeat(self) -> RuntimeConfig:
        if self.heartbeat_seconds >= self.lease_ttl_seconds:
            raise ValueError("heartbeat_seconds must be less than lease_ttl_seconds")
        return self


class AdvancedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_max_input: int | None = Field(default=None, gt=0)
    generation_max_output: int | None = Field(default=None, gt=0)
    embedding_max_text_length: int | None = Field(default=None, gt=0)
    retry_attempts: int = Field(default=5, ge=0, le=10)
    allow_custom_model: bool = False
    custom_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    custom_runtime_compatibility: str | None = None


class IntegrationConfig(BaseModel):
    """One explicitly selected agent integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["hermes", "codex", "claude-code", "cursor", "opencode", "openclaw"]
    enabled: bool = True


class InstallerConfig(BaseModel):
    """The single schema shared by the wizard and agent-driven installs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    integrations: tuple[IntegrationConfig, ...] = ()
    # Installation must make the semantic language an explicit deployment
    # choice; it is never inferred from user text or the host locale.
    semantic_language: Literal["ru", "en", "es", "pt", "fr", "de", "uk"]
    # Agents may either share one logical knowledge space or keep independent
    # spaces while still using the same Local/Core runtime.
    memory_mode: Literal["shared", "per_agent"] = "per_agent"
    memory_data_path: str | None = None
    generation: GenerationConfig
    embedding: EmbeddingConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("schema_version") == 1 or "target" in payload:
            target = payload.pop("target", "hermes")
            payload["schema_version"] = 2
            payload.setdefault("integrations", [{"id": target, "enabled": True}])
        return payload

    @model_validator(mode="after")
    def validate_integrations(self) -> InstallerConfig:
        identifiers = [item.id for item in self.integrations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("integrations must contain unique ids")
        return self

    def memory_space_id_for(self, target_id: str) -> str:
        """Return the logical memory space injected into an agent adapter."""

        if self.memory_mode == "shared":
            return "shared-default"
        return f"{target_id}-default"

    @field_validator("memory_data_path")
    @classmethod
    def validate_memory_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _text(value, "memory_data_path")

    @model_validator(mode="after")
    def validate_custom_local_model(self) -> InstallerConfig:
        local = self.embedding.local
        if local is None or local.model_path is None:
            return self
        if not self.advanced.allow_custom_model:
            raise ValueError(
                "local model_path is available only in explicit advanced mode"
            )
        if not self.advanced.custom_model_sha256:
            raise ValueError("advanced custom model requires custom_model_sha256")
        if not self.advanced.custom_runtime_compatibility:
            raise ValueError(
                "advanced custom model requires runtime compatibility metadata"
            )
        return self


class ProfileBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    slot: Literal["operational", "background", "embedding"]
    model: str
    endpoint: str


__all__ = [
    "AdvancedConfig",
    "EmbeddingApiConfig",
    "EmbeddingConfig",
    "GenerationConfig",
    "InstallerConfig",
    "IntegrationConfig",
    "LocalEmbeddingConfig",
    "ProfileBinding",
    "RuntimeConfig",
]
