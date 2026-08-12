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
    generation_profile_fingerprint,
)
from .providers.base import (
    ChatMessage,
    InferenceProvider,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
    ProviderResponseError,
    normalize_error,
    normalize_usage,
)
from .providers.openai_compatible import (
    DEFAULT_TOOL_NAME,
    OpenAICompatibleProvider,
    decode_json_content,
)
from .providers.google_boundary import GoogleGenerationTransport
from .secrets import SecretNotFoundError, SecretStore
from .token_budget import (
    InputBudgetExceededError,
    OutputBudgetExceededError,
    TokenBudgetEstimator,
)

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

    if profile.provider_kind == "google_genai":
        del secret
        return GoogleGenerationTransport()
    if profile.provider_kind == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=profile.base_url,
            api_key=secret,
            timeout_seconds=profile.timeout_seconds,
            max_retries=profile.max_retries,
            token_parameter=profile.token_parameter,
            supports_system_role=profile.supports_system_role,
            supports_seed=profile.supports_seed,
        )
    raise ProviderConfigurationError("unsupported inference provider kind")


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
    normalized_usage: dict[str, object] = Field(default_factory=dict)

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

    @property
    def parsed_json(self) -> dict[str, object]:
        """Canonical name for the parseable payload sent to Core."""

        return self.data

    @property
    def usage(self) -> dict[str, object] | None:
        """Provider usage under the normalized response vocabulary."""

        return self.token_usage

    @property
    def provider_request_id(self) -> str | None:
        value = self.metadata.get("provider_request_id", self.metadata.get("request_id"))
        return value if isinstance(value, str) and value else None

    @property
    def finish_reason(self) -> str | None:
        value = self.metadata.get("finish_reason")
        return value if isinstance(value, str) and value else None

    @property
    def transport_error(self) -> object | None:
        """Successful normalized responses carry an explicit null error."""

        return self.metadata.get("transport_error")

    @property
    def native_schema_issues(self) -> list[dict[str, object]]:
        value = self.metadata.get("local_validation_issues")
        return value if isinstance(value, list) else []

    @property
    def native_schema_valid(self) -> bool:
        return not self.native_schema_issues


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
        if max_output_tokens > profile.max_output_tokens:
            error = OutputBudgetExceededError(
                max_output_tokens,
                profile.max_output_tokens,
            )
            error.profile_id = profile.profile_id
            raise error
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
        capabilities = self._load_capabilities(profile)
        try:
            execute = getattr(provider, "execute_structured", None)
            if callable(execute):
                response = execute(
                    request,
                    profile,
                    capabilities,
                    cancellation_token=cancellation_token,
                )
            else:
                response = provider.complete_json(
                    request, cancellation_token=cancellation_token
                )
        except Exception as exc:
            self._invalidate_capability_after_failure(
                profile,
                selected_mode=selected_mode,
                error=exc,
            )
            raise
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
            profile_getter = getattr(store, "get_capabilities_for_profile", None)
            if callable(profile_getter):
                result = profile_getter(profile, fresh_only=True)
                if isinstance(result, ProviderCapabilities):
                    return result
            getter = getattr(store, "get_capabilities", None)
            if callable(getter):
                result = getter(profile.profile_id)
                if isinstance(result, ProviderCapabilities):
                    fingerprint = generation_profile_fingerprint(profile)
                    if result.profile_fingerprint and not result.is_fresh(
                        profile_fingerprint=fingerprint
                    ):
                        continue
                    return result
        database_path = getattr(self._profile_resolver, "database_path", None)
        if database_path is not None:
            from .profile_store import DatabaseBackedCapabilityStore

            store = DatabaseBackedCapabilityStore(str(database_path))
            getter = getattr(store, "get_capabilities_for_profile", None)
            if callable(getter):
                return getter(profile, fresh_only=True)
            return store.get_capabilities(profile.profile_id)
        return None

    def _invalidate_capability_after_failure(
        self,
        profile: InferenceProfile,
        *,
        selected_mode: StructuredOutputMode,
        error: BaseException,
    ) -> None:
        """Invalidate only the affected cache entry after a format failure."""

        # Transport outages, authentication failures and timeouts say nothing
        # about the previously probed structured-output mode. Retain a still
        # fresh capability observation and let the normal retry/outage policy
        # report the execution failure. Only an actual provider response
        # incompatibility can invalidate the selected mode.
        if selected_mode == "auto" or not isinstance(
            error, (ProviderResponseError, StructuredJsonResponseError)
        ):
            return
        reason = f"capability_execution_failed:{selected_mode}:{normalize_error(error)['code']}"
        stores: list[object] = []
        if self._capability_store is not None:
            stores.append(self._capability_store)
        resolver_store = getattr(self._profile_resolver, "profile_store", None)
        if resolver_store is not None:
            stores.append(resolver_store)
        for store in stores:
            invalidator = getattr(store, "invalidate_capabilities", None)
            if callable(invalidator):
                invalidator(profile.profile_id, reason=reason)
                return

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
        normalized_usage = normalize_usage(response)
        metadata_value["normalized_usage"] = normalized_usage
        metadata_value["parsed_json"] = parsed
        metadata_value["raw_text"] = response.raw_text or response.content
        provider_request_id = metadata_value.get("request_id")
        if isinstance(provider_request_id, str) and provider_request_id:
            metadata_value["provider_request_id"] = provider_request_id
        finish_reason = metadata_value.get("finish_reason")
        if not isinstance(finish_reason, str):
            metadata_value["finish_reason"] = None
        metadata_value["transport_error"] = None
        metadata_value["local_validation_issues"] = _advisory_schema_issues(
            parsed,
            output_contract.get("json_schema")
            if isinstance(output_contract, Mapping)
            else None,
        )
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
            normalized_usage=normalized_usage,
        )


def _contract_digest(contract: Mapping[str, object] | None) -> str | None:
    if contract is None:
        return None
    digest = contract.get("schema_digest")
    if digest is None:
        digest = contract.get("contract_digest")
    return digest if isinstance(digest, str) and digest else None


def _advisory_schema_issues(
    value: object,
    schema: object,
    *,
    path: str = "$",
    limit: int = 16,
) -> list[dict[str, object]]:
    """Return bounded diagnostics without making schema mismatch terminal.

    Core owns the authoritative schema/repair/semantic decision.  Local only
    records a small, provider-neutral advisory report so Lab can distinguish
    native schema compliance from a parseable response that Core can repair.
    """

    if not isinstance(schema, Mapping):
        return []
    issues: list[dict[str, object]] = []

    def add(issue_path: str, code: str, message: str) -> None:
        if len(issues) < limit:
            issues.append({"path": issue_path, "code": code, "message": message})

    schema_type = schema.get("type")
    if schema_type == "object" and not isinstance(value, Mapping):
        add(path, "type", "expected object")
        return issues
    if schema_type == "array" and not isinstance(value, list):
        add(path, "type", "expected array")
        return issues
    if schema_type == "string" and not isinstance(value, str):
        add(path, "type", "expected string")
        return issues
    if schema_type == "boolean" and not isinstance(value, bool):
        add(path, "type", "expected boolean")
        return issues
    if "const" in schema and value != schema.get("const"):
        add(path, "const", "value does not match const")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        add(path, "enum", "value is not in enum")
    if isinstance(value, Mapping):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    add(f"{path}.{key}", "required", "required field is missing")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if isinstance(key, str) and key in value:
                    issues.extend(
                        _advisory_schema_issues(
                            value[key],
                            child_schema,
                            path=f"{path}.{key}",
                            limit=max(0, limit - len(issues)),
                        )
                    )
                    if len(issues) >= limit:
                        break
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                issues.extend(
                    _advisory_schema_issues(
                        item,
                        item_schema,
                        path=f"{path}[{index}]",
                        limit=max(0, limit - len(issues)),
                    )
                )
                if len(issues) >= limit:
                    break
    return issues[:limit]


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
