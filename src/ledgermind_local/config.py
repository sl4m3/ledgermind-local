"""Local service configuration models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class LocalConfig(BaseModel):
    """Strict service configuration for the local LedgerMind process."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(ge=1)
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8765, ge=1, le=65535)
    database_path: str = "ledgermind.db"
    log_level: str = "INFO"
    projection_poll_interval_seconds: float = Field(default=1.0, ge=0.0)
    allow_remote_bind: bool = False
    vector: VectorProjectionConfig = Field(default_factory=VectorProjectionConfig)
    markdown_projection: MarkdownProjectionConfig = Field(default_factory=MarkdownProjectionConfig)
    markdown_audit: GitAuditConfig = Field(default_factory=GitAuditConfig)
    markdown_audit_enabled: bool = False

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

    @field_validator("allow_remote_bind", mode="after")
    @classmethod
    def _validate_remote_bind(cls, value: bool, info: Any) -> bool:
        # pylint: disable=unused-argument
        if not value and info.data.get("bind_host") not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("bind_host outside localhost requires allow_remote_bind=true")
        return value

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
        return cls.model_validate_json(payload)

    @classmethod
    def from_file(cls, path: Path) -> LocalConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, payload: object) -> LocalConfig:
        return cls.model_validate(payload)

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
