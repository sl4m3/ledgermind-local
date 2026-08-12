"""Structured generation over Local-owned profiles and capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .cancellation import CancellationToken
from .profile_slots import ProfileResolver, ProfileSlot
from .profiles import (
    InferenceProfile,
    ProviderCapabilities,
    StructuredOutputMode,
)
from .providers.base import (
    ChatMessage,
    InferenceProvider,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
)
from .providers.openai_compatible import (
    DEFAULT_TOOL_NAME,
    OpenAICompatibleProvider,
    decode_json_content,
)
from .secrets import SecretNotFoundError, SecretStore
from .token_budget import InputBudgetExceededError, TokenBudgetEstimator

ProviderFactory = Callable[[InferenceProfile, str], InferenceProvider]
_MODE_ORDER: tuple[StructuredOutputMode, ...] = (
    "json_schema",
    "tool_call",
    "json_object",
    "prompt_only",
)


class CapabilityStore(Protocol):
    def get_capabilities(self, profile_id: str) -> ProviderCapabilities | None: ...


def default_provider_factory(profile: InferenceProfile, secret: str) -> InferenceProvider:
    """Construct the default generative provider for a resolved profile."""

    if profile.provider_kind != "openai_compatible":
        raise ProviderConfigurationError("unsupported inference provider kind")
    return OpenAICompatibleProvider(
        base_url=profile.base_url,
        api_key=secret,
        timeout_seconds=profile.timeout_seconds,
        max_retries=profile.max_retries,
        token_parameter=profile.token_parameter,
        supports_system_role=profile.supports_system_role,
        supports_seed=profile.supports_seed,
    )


class StructuredJsonResult(BaseModel):
    """Parsed JSON object plus selected transport mode and safe metadata."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    data: dict[str, object]
    raw_text: str = ""
    profile_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    status_code: int = Field(ge=100, le=599)
    structured_output_mode: StructuredOutputMode = Field(
        default="json_object",
        validation_alias=AliasChoices("structured_output_mode", "mode"),
    )
    contract_digest: str | None = Field(default=None, max_length=200)
    tool_name: str | None = Field(default=None, min_length=1, max_length=200)
    metadata: dict[str, object] = Field(default_factory=dict)
    token_usage: dict[str, object] | None = None

    @property
    def mode(self) -> StructuredOutputMode:
        return self.structured_output_mode

    @property
    def selected_mode(self) -> StructuredOutputMode:
        return self.structured_output_mode

    @property
    def provider_metadata(self) -> dict[str, object]:
        return self.metadata

    @property
    def raw_model_text(self) -> str:
        return self.raw_text


class StructuredJsonError(RuntimeError):
    """Base structured generation failure with a safe code."""

    code = "structured_json_error"


class StructuredJsonRequestError(StructuredJsonError):
    code = "invalid_request"


class StructuredJsonSecretError(StructuredJsonError):
    code = "secret_missing"


class StructuredJsonResponseError(StructuredJsonError):
    code = "invalid_json_response"


class StructuredJsonProvider:
    """Select a technical provider mode and return a Core-facing JSON object.

    This class checks only transport shape (object, JSON, size, and capability
    metadata).  It never assigns or interprets operation, facet, resolution, or
    value semantics.
    """

    def __init__(
        self,
        *,
        profile_resolver: ProfileResolver,
        secret_store: SecretStore,
        provider_factory: ProviderFactory | None = None,
        capability_store: CapabilityStore | None = None,
        capabilities_store: CapabilityStore | None = None,
        max_output_bytes: int = 2_000_000,
        token_budget_estimator: TokenBudgetEstimator | None = None,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        if capability_store is not None and capabilities_store is not None:
            raise ValueError("provide only one capability store")
        self._profile_resolver = profile_resolver
        self._secret_store = secret_store
        self._provider_factory = provider_factory or default_provider_factory
        self._capability_store = capability_store or capabilities_store
        self.max_output_bytes = max_output_bytes
        self._token_budget_estimator = token_budget_estimator or TokenBudgetEstimator()

    def generate_json(
        self,
        *,
        memory_space_id: str,
        messages: Sequence[ChatMessage],
        max_output_tokens: int,
        profile_slot: ProfileSlot,
        response_format: Mapping[str, object] | None = None,
        output_contract: Mapping[str, object] | None = None,
        mode: StructuredOutputMode | None = None,
        structured_output_mode: StructuredOutputMode | None = None,
        tool_name: str | None = None,
        metadata: Mapping[str, object] | None = None,
        seed: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> StructuredJsonResult:
        profile = self._profile_resolver.resolve_profile(memory_space_id, profile_slot)
        contract, requested_mode = self._normalize_contract_and_mode(
            response_format=response_format,
            output_contract=output_contract,
            mode=mode,
            structured_output_mode=structured_output_mode,
        )
        selected_mode = self._select_mode(
            profile,
            requested_mode=requested_mode,
            response_format=response_format,
        )
        request = self._build_request(
            profile=profile,
            messages=messages,
            max_output_tokens=max_output_tokens,
            output_contract=contract,
            mode=selected_mode,
            tool_name=tool_name,
            metadata=metadata,
            seed=seed,
        )
        # This is deliberately before secret lookup/provider construction and
        # therefore before any possible HTTP call.
        try:
            self._token_budget_estimator.ensure_within(
                request, profile.max_input_tokens
            )
        except InputBudgetExceededError as exc:
            exc.profile_id = profile.profile_id
            raise
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
        return self._to_result(
            profile,
            response,
            output_contract=contract,
            selected_mode=selected_mode,
            tool_name=tool_name,
        )

    def _normalize_contract_and_mode(
        self,
        *,
        response_format: Mapping[str, object] | None,
        output_contract: Mapping[str, object] | None,
        mode: StructuredOutputMode | None,
        structured_output_mode: StructuredOutputMode | None,
    ) -> tuple[dict[str, object] | None, StructuredOutputMode | None]:
        if mode is not None and structured_output_mode is not None and mode != structured_output_mode:
            raise StructuredJsonRequestError("conflicting structured output modes")
        selected_request_mode = mode or structured_output_mode
        contract = dict(output_contract) if output_contract is not None else None
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                # Explicit legacy response_format always wins over profile
                # auto/manual defaults, preserving the old API behavior.
                selected_request_mode = "json_object"
            elif format_type == "json_schema":
                selected_request_mode = "json_schema"
                schema_contract = response_format.get("json_schema")
                if isinstance(schema_contract, dict):
                    if isinstance(schema_contract.get("schema"), dict):
                        contract = {
                            "contract_name": schema_contract.get(
                                "name", "structured_result"
                            ),
                            "json_schema": dict(schema_contract["schema"]),
                        }
                    else:
                        contract = dict(schema_contract)
            else:
                raise StructuredJsonRequestError(
                    "response_format must be json_object or json_schema"
                )
        return contract, selected_request_mode

    def _load_capabilities(self, profile: InferenceProfile) -> ProviderCapabilities | None:
        stores: list[object] = []
        if self._capability_store is not None:
            stores.append(self._capability_store)
        resolver_store = getattr(self._profile_resolver, "profile_store", None)
        if resolver_store is not None:
            stores.append(resolver_store)
        for store in stores:
            getter = getattr(store, "get_capabilities", None)
            if callable(getter):
                result = getter(profile.profile_id)
                if isinstance(result, ProviderCapabilities):
                    return result
        database_path = getattr(self._profile_resolver, "database_path", None)
        if database_path is not None:
            from .profile_store import DatabaseBackedCapabilityStore

            return DatabaseBackedCapabilityStore(str(database_path)).get_capabilities(
                profile.profile_id
            )
        return None

    def _select_mode(
        self,
        profile: InferenceProfile,
        *,
        requested_mode: StructuredOutputMode | None,
        response_format: Mapping[str, object] | None,
    ) -> StructuredOutputMode:
        if response_format is not None and response_format.get("type") == "json_object":
            return "json_object"
        candidate = requested_mode or profile.structured_output_preference
        capabilities = self._load_capabilities(profile)
        if candidate != "auto":
            # A non-auto preference is an explicit operator choice.  It is
            # sent as requested; an endpoint rejection remains observable as
            # a provider error rather than being silently changed to another
            # mode.
            return candidate
        if capabilities is not None:
            for mode in _MODE_ORDER:
                if capabilities.supports(mode):
                    return mode
            if capabilities.probe_status == "passed":
                raise StructuredJsonRequestError(
                    "provider has no persisted structured output capability"
                )
        # A profile created before capability probing must retain the old
        # json_object behavior until a probe records a better mode.
        return "json_object"

    def _build_request(
        self,
        *,
        profile: InferenceProfile,
        messages: Sequence[ChatMessage],
        max_output_tokens: int,
        output_contract: dict[str, object] | None,
        mode: StructuredOutputMode,
        tool_name: str | None,
        metadata: Mapping[str, object] | None,
        seed: int | None,
    ) -> ModelRequest:
        if not messages:
            raise StructuredJsonRequestError("messages must not be empty")
        if mode == "json_schema" and (
            not isinstance(output_contract, dict)
            or not isinstance(output_contract.get("json_schema"), dict)
        ):
            raise StructuredJsonRequestError(
                "json_schema mode requires an output contract schema"
            )
        try:
            return ModelRequest(
                model=profile.model,
                messages=tuple(messages),
                max_output_tokens=max_output_tokens,
                output_contract=output_contract,
                mode=mode,
                tool_name=tool_name,
                metadata=dict(metadata or {}),
                token_parameter=profile.token_parameter,
                supports_system_role=profile.supports_system_role,
                supports_seed=profile.supports_seed,
                seed=seed,
            )
        except ValueError as exc:
            raise StructuredJsonRequestError("model request is invalid") from exc

    def _to_result(
        self,
        profile: InferenceProfile,
        response: ModelResponse,
        *,
        output_contract: dict[str, object] | None,
        selected_mode: StructuredOutputMode,
        tool_name: str | None,
    ) -> StructuredJsonResult:
        if response.response_bytes > self.max_output_bytes:
            raise StructuredJsonResponseError("provider response exceeds size limit")
        try:
            parsed = decode_json_content(response.content)
        except Exception as exc:
            if isinstance(exc, StructuredJsonResponseError):
                raise
            raise StructuredJsonResponseError(
                "provider response is not valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise StructuredJsonResponseError(
                "provider response must be a JSON object"
            )
        metadata_value = dict(response.metadata)
        token_usage = metadata_value.get("usage")
        return StructuredJsonResult(
            data=parsed,
            raw_text=response.raw_text or response.content,
            profile_id=profile.profile_id,
            provider=profile.provider_kind,
            model=response.model,
            attempts=response.attempts,
            request_bytes=response.request_bytes,
            response_bytes=response.response_bytes,
            status_code=response.status_code,
            structured_output_mode=selected_mode,
            contract_digest=_contract_digest(output_contract),
            tool_name=(
                tool_name
                or response.tool_name
                or (DEFAULT_TOOL_NAME if selected_mode == "tool_call" else None)
            ),
            metadata=metadata_value,
            token_usage=token_usage if isinstance(token_usage, dict) else None,
        )


def _contract_digest(contract: Mapping[str, object] | None) -> str | None:
    if contract is None:
        return None
    digest = contract.get("schema_digest")
    if digest is None:
        digest = contract.get("contract_digest")
    return digest if isinstance(digest, str) and digest else None


__all__ = [
    "CapabilityStore",
    "InputBudgetExceededError",
    "ProviderFactory",
    "StructuredJsonError",
    "StructuredJsonProvider",
    "StructuredJsonRequestError",
    "StructuredJsonResponseError",
    "StructuredJsonResult",
    "StructuredJsonSecretError",
    "default_provider_factory",
]
