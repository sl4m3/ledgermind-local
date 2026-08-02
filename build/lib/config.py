"""Local service configuration models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LocalConfig(BaseModel):
    """Strict service configuration for the local LedgerMind process."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(ge=1)
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8765, ge=1, le=65535)
    allow_remote_bind: bool = False

    @field_validator("bind_host")
    @classmethod
    def _validate_bind_host(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("bind_host must not be empty")
        return stripped

    @field_validator("allow_remote_bind", mode="after")
    @classmethod
    def _validate_remote_bind(cls, value: bool, info):
        # pylint: disable=unused-argument
        if not value and info.data.get("bind_host") not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("bind_host outside localhost requires allow_remote_bind=true")
        return value

    @classmethod
    def from_json(cls, payload: str) -> "LocalConfig":
        return cls.model_validate_json(payload)

    @classmethod
    def from_file(cls, path: Path) -> "LocalConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, payload: object) -> "LocalConfig":
        return cls.model_validate(payload)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
