"""Inference profile models owned by the Local service."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderKind = Literal["openai_compatible"]


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class InferenceProfile(BaseModel):
    """Strict, secret-free configuration for one model provider profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=200)
    provider_kind: ProviderKind = "openai_compatible"
    base_url: str = Field(min_length=1, max_length=2_000)
    model: str = Field(min_length=1, max_length=500)
    secret_ref: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_input_tokens: int = Field(default=12_000, ge=1, le=200_000)
    max_output_tokens: int = Field(default=2_000, ge=1, le=50_000)
    hypothesis_prompt_version: int = Field(default=1, ge=1)
    hypothesis_schema_version: int = Field(default=1, ge=1)
    merge_prompt_version: int = Field(default=1, ge=1)
    merge_schema_version: int = Field(default=1, ge=1)
    enabled: bool = True

    @field_validator("profile_id", "model", "secret_ref")
    @classmethod
    def _validate_text_fields(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _required_text(value, field_name)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        normalized = _required_text(value, "base_url").rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        return normalized


class MemorySpaceInferenceProfiles(BaseModel):
    """Profile bindings for one memory space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_space_id: str = Field(min_length=1, max_length=200)
    hypothesis_profile_id: str | None = Field(default=None, max_length=200)
    merge_profile_id: str | None = Field(default=None, max_length=200)

    @field_validator("memory_space_id", "hypothesis_profile_id", "merge_profile_id")
    @classmethod
    def _normalize_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "value")
        return _required_text(value, field_name)


__all__ = ["InferenceProfile", "MemorySpaceInferenceProfiles", "ProviderKind"]
