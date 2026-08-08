"""Structured JSON completion provider over the existing generative provider."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cancellation import CancellationToken
from .profile_slots import ProfileResolver, ProfileSlot
from .profiles import InferenceProfile
from .providers.base import (
    ChatMessage,
    InferenceProvider,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
)
from .providers.openai_compatible import OpenAICompatibleProvider
from .secrets import SecretNotFoundError, SecretStore

ProviderFactory = Callable[[InferenceProfile, str], InferenceProvider]


def default_provider_factory(profile: InferenceProfile, secret: str) -> InferenceProvider:
    """Construct the default generative provider for a resolved profile."""
    if profile.provider_kind != "openai_compatible":
        raise ProviderConfigurationError("unsupported inference provider kind")
    return OpenAICompatibleProvider(
        base_url=profile.base_url,
        api_key=secret,
        timeout_seconds=profile.timeout_seconds,
        max_retries=profile.max_retries,
    )


class StructuredJsonResult(BaseModel):
    """Typed JSON object result with provider metadata and safe byte counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: dict[str, object]
    profile_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    status_code: int = Field(ge=100, le=599)
    token_usage: dict[str, object] | None = None


class StructuredJsonError(RuntimeError):
    """Base structured JSON failure with a safe code."""

    code = "structured_json_error"


class StructuredJsonRequestError(StructuredJsonError):
    code = "invalid_request"


class StructuredJsonSecretError(StructuredJsonError):
    code = "secret_missing"


class StructuredJsonResponseError(StructuredJsonError):
    code = "invalid_json_response"


class StructuredJsonProvider:
    """Generic JSON-object completion through a generative inference provider.

    Validates only transport-level and envelope-level properties: the response
    is a JSON object, bounded in size, with provider status. It does not
    interpret facet, object-id, or merge semantics.
    """

    def __init__(
        self,
        *,
        profile_resolver: ProfileResolver,
        secret_store: SecretStore,
        provider_factory: ProviderFactory | None = None,
        max_output_bytes: int = 2_000_000,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self._profile_resolver = profile_resolver
        self._secret_store = secret_store
        self._provider_factory = provider_factory or default_provider_factory
        self.max_output_bytes = max_output_bytes

    def generate_json(
        self,
        *,
        memory_space_id: str,
        messages: Sequence[ChatMessage],
        max_output_tokens: int,
        response_format: Mapping[str, object] | None,
        profile_slot: ProfileSlot,
        cancellation_token: CancellationToken | None = None,
    ) -> StructuredJsonResult:
        profile = self._profile_resolver.resolve_profile(memory_space_id, profile_slot)
        request = self._build_request(
            profile=profile,
            messages=messages,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )
        try:
            secret = self._secret_store.get(profile.secret_ref)
        except SecretNotFoundError as exc:
            raise StructuredJsonSecretError(
                "configured provider secret is not present"
            ) from exc

        provider = self._provider_factory(profile, secret)
        try:
            response = provider.complete_json(
                request, cancellation_token=cancellation_token
            )
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        return self._to_result(profile, response)

    def _build_request(
        self,
        *,
        profile: InferenceProfile,
        messages: Sequence[ChatMessage],
        max_output_tokens: int,
        response_format: Mapping[str, object] | None,
    ) -> ModelRequest:
        if not messages:
            raise StructuredJsonRequestError("messages must not be empty")
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type != "json_object":
                raise StructuredJsonRequestError(
                    "only the json_object response format is supported"
                )
        try:
            return ModelRequest(
                model=profile.model,
                messages=tuple(messages),
                max_output_tokens=max_output_tokens,
            )
        except ValueError as exc:
            raise StructuredJsonRequestError("model request is invalid") from exc

    def _to_result(
        self, profile: InferenceProfile, response: ModelResponse
    ) -> StructuredJsonResult:
        if response.response_bytes > self.max_output_bytes:
            raise StructuredJsonResponseError("provider response exceeds size limit")
        try:
            parsed = json.loads(response.content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise StructuredJsonResponseError(
                "provider response is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise StructuredJsonResponseError(
                "provider response must be a JSON object"
            )
        return StructuredJsonResult(
            data=parsed,
            profile_id=profile.profile_id,
            provider=profile.provider_kind,
            model=response.model,
            attempts=response.attempts,
            request_bytes=response.request_bytes,
            response_bytes=response.response_bytes,
            status_code=response.status_code,
        )


__all__ = [
    "ProviderFactory",
    "StructuredJsonError",
    "StructuredJsonProvider",
    "StructuredJsonRequestError",
    "StructuredJsonResponseError",
    "StructuredJsonResult",
    "StructuredJsonSecretError",
    "default_provider_factory",
]
