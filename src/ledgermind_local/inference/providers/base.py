"""Typed provider boundary for Local inference calls."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ..cancellation import CancellationToken
from ..profiles import StructuredOutputMode, TokenParameter


class ChatMessage(BaseModel):
    """One bounded chat message sent to a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelRequest(BaseModel):
    """Strict structured-completion request; never an arbitrary mapping."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    model: str = Field(min_length=1, max_length=500)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=32)
    max_output_tokens: int = Field(gt=0, le=50_000)
    response_format: dict[str, object] | None = None
    output_contract: dict[str, object] | None = None
    mode: StructuredOutputMode = Field(
        default="json_object",
        validation_alias=AliasChoices("mode", "structured_output_mode"),
    )
    tool_name: str | None = Field(default=None, min_length=1, max_length=200)
    metadata: dict[str, object] = Field(default_factory=dict)
    token_parameter: TokenParameter | None = None
    supports_system_role: bool | None = None
    supports_seed: bool | None = None
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)

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
        response_format: dict[str, object] | None = None,
        output_contract: dict[str, object] | None = None,
        mode: StructuredOutputMode = "json_object",
        tool_name: str | None = None,
        metadata: dict[str, object] | None = None,
        token_parameter: TokenParameter | None = None,
        supports_system_role: bool | None = None,
        supports_seed: bool | None = None,
        seed: int | None = None,
    ) -> ModelRequest:
        return cls(
            model=model,
            messages=(
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ),
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            output_contract=output_contract,
            mode=mode,
            tool_name=tool_name,
            metadata=metadata or {},
            token_parameter=token_parameter,
            supports_system_role=supports_system_role,
            supports_seed=supports_seed,
            seed=seed,
        )

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        """Compatibility name used by the Core-facing task contract."""

        return self.mode

    def to_openai_payload(self) -> dict[str, object]:
        from .openai_compatible import build_payload

        return build_payload(self)

    def encoded_payload(self) -> bytes:
        return json.dumps(
            self.to_openai_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class ModelResponse(BaseModel):
    """Provider response content and secret-free transport metadata."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    content: str = Field(min_length=2)
    model: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    status_code: int = Field(ge=100, le=599)
    output_contract: dict[str, object] | None = None
    mode: StructuredOutputMode = Field(
        default="json_object",
        validation_alias=AliasChoices("mode", "structured_output_mode"),
    )
    tool_name: str | None = Field(default=None, min_length=1, max_length=200)
    metadata: dict[str, object] = Field(default_factory=dict)
    raw_text: str | None = None

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        return self.mode

    @property
    def selected_mode(self) -> StructuredOutputMode:
        return self.mode

    @property
    def provider_metadata(self) -> dict[str, object]:
        return self.metadata

    @property
    def raw_model_text(self) -> str:
        return self.raw_text or self.content


class ProviderError(RuntimeError):
    """Base error with safe, non-payload diagnostics."""

    code = "provider_error"


class ProviderConfigurationError(ProviderError):
    code = "configuration_error"


class ProviderAuthenticationError(ProviderError):
    code = "authentication_error"


class ProviderTimeoutError(ProviderError):
    code = "timeout"


class ProviderCancelledError(ProviderError):
    """Safe provider cancellation signal without request or payload details."""

    code = "cancelled"

    def __init__(self, *args: object) -> None:
        del args
        super().__init__("provider request cancelled")


class ProviderTransportError(ProviderError):
    code = "transport_error"


class TransientProviderError(ProviderError):
    code = "transient_provider_error"


class ProviderResponseError(ProviderError):
    code = "invalid_provider_response"


class InferenceProvider(Protocol):
    """Structural interface implemented by concrete model providers."""

    provider_kind: str = "unknown"

    def complete_json(
        self,
        request: ModelRequest,
        token: CancellationToken | None = None,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ModelResponse: ...


def messages_as_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Return a serialization helper without exposing Pydantic internals."""

    return [{"role": message.role, "content": message.content} for message in messages]


__all__ = [
    "ChatMessage",
    "InferenceProvider",
    "ModelRequest",
    "ModelResponse",
    "ProviderAuthenticationError",
    "ProviderCancelledError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "StructuredOutputMode",
    "TokenParameter",
    "TransientProviderError",
]
