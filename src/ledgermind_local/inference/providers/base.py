"""Typed provider boundary for Local inference calls."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """One bounded chat message sent to a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelRequest(BaseModel):
    """Strict JSON-completion request; never an arbitrary mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=500)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=32)
    max_output_tokens: int = Field(gt=0, le=50_000)

    @field_validator("model")
    @classmethod
    def _normalize_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model must not be empty")
        return normalized

    @classmethod
    def from_messages(
        cls,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> ModelRequest:
        return cls(
            model=model,
            messages=(
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ),
            max_output_tokens=max_output_tokens,
        )

    def to_openai_payload(self) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [message.model_dump() for message in self.messages],
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

    def encoded_payload(self) -> bytes:
        return json.dumps(
            self.to_openai_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class ModelResponse(BaseModel):
    """Provider response metadata and JSON content without raw payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=2)
    model: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    status_code: int = Field(ge=100, le=599)


class ProviderError(RuntimeError):
    """Base error with safe, non-payload diagnostics."""

    code = "provider_error"


class ProviderConfigurationError(ProviderError):
    code = "configuration_error"


class ProviderAuthenticationError(ProviderError):
    code = "authentication_error"


class ProviderTimeoutError(ProviderError):
    code = "timeout"


class ProviderTransportError(ProviderError):
    code = "transport_error"


class TransientProviderError(ProviderError):
    code = "transient_provider_error"


class ProviderResponseError(ProviderError):
    code = "invalid_provider_response"


class InferenceProvider(Protocol):
    """Structural interface implemented by concrete model providers."""

    provider_kind: str = "unknown"

    def complete_json(self, request: ModelRequest) -> ModelResponse: ...


def messages_as_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Return a serialization helper without exposing Pydantic internals."""

    return [{"role": message.role, "content": message.content} for message in messages]


__all__ = [
    "ChatMessage",
    "InferenceProvider",
    "ModelRequest",
    "ModelResponse",
    "ProviderAuthenticationError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "TransientProviderError",
]
