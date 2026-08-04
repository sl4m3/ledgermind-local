"""Local inference broker coordinating profiles, providers, and typed output."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..persistence import open_sqlite_connection
from ..persistence import rounds_migrations as migrations
from ..processing.generator import HypothesisCandidate
from ..processing.models import NormalizedRound
from .profile_store import InferenceProfileStore
from .profiles import InferenceProfile
from .providers.base import InferenceProvider, ModelRequest, ModelResponse
from .providers.openai_compatible import OpenAICompatibleProvider
from .schemas import HypothesisResponse, MergeProposal
from .secrets import SecretStore


class InferenceBrokerError(RuntimeError):
    """Base safe broker error."""

    code = "inference_broker_error"


class InferenceProfileNotFoundError(InferenceBrokerError):
    code = "profile_not_found"


class InferenceProfileDisabledError(InferenceBrokerError):
    code = "profile_disabled"


class InferenceInputTooLargeError(InferenceBrokerError):
    code = "input_too_large"


class InferenceResponseValidationError(InferenceBrokerError):
    code = "response_validation_error"


class ModelTask(BaseModel):
    """Strict provider task passed through the broker boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_space_id: str = Field(min_length=1, max_length=200)
    operation: Literal["hypothesis", "merge"]
    request: ModelRequest


ProviderFactory = Callable[[InferenceProfile, str], InferenceProvider]


def _hypothesis_prompt() -> str:
    try:
        return (
            resources.files("ledgermind_local.inference")
            .joinpath("prompts/hypothesis_v1.txt")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, UnicodeError) as exc:
        raise InferenceBrokerError("hypothesis prompt version is unavailable") from exc


def _merge_prompt() -> str:
    try:
        return (
            resources.files("ledgermind_local.inference")
            .joinpath("prompts/merge_v1.txt")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, UnicodeError) as exc:
        raise InferenceBrokerError("merge prompt version is unavailable") from exc


def _normalized_round_payload(normalized_round: NormalizedRound) -> dict[str, object]:
    return {
        "source_round_id": normalized_round.source_round_id,
        "source_event_ids": list(normalized_round.source_event_ids),
        "user_text": normalized_round.user_text,
        "assistant_text": normalized_round.assistant_text,
        "transcript": normalized_round.transcript,
        "tool_interactions": [
            {
                "tool_call_id": interaction.tool_call_id,
                "tool_name": interaction.tool_name,
                "arguments_json": interaction.arguments_json,
                "result_text": interaction.result_text,
                "result_json": interaction.result_json,
                "status": interaction.status,
                "error_text": interaction.error_text,
                "source_call_event_id": interaction.source_call_event_id,
                "source_result_event_id": interaction.source_result_event_id,
            }
            for interaction in normalized_round.tool_interactions
        ],
    }


class InferenceBroker:
    """Only Local-owned component allowed to call an inference provider."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        secret_store: SecretStore,
        provider_factory: ProviderFactory | None = None,
        max_input_chars: int | None = None,
    ) -> None:
        self.database_path = database_path
        self.secret_store = secret_store
        self.provider_factory = provider_factory or self._default_provider
        self.max_input_chars = max_input_chars
        if max_input_chars is not None and max_input_chars < 1:
            raise ValueError("max_input_chars must be positive")

    @staticmethod
    def _default_provider(profile: InferenceProfile, secret: str) -> InferenceProvider:
        if profile.provider_kind != "openai_compatible":
            raise InferenceBrokerError("unsupported inference provider kind")
        return OpenAICompatibleProvider(
            base_url=profile.base_url,
            api_key=secret,
            timeout_seconds=profile.timeout_seconds,
            max_retries=profile.max_retries,
        )

    def _load_profile(self, profile_id: str) -> InferenceProfile:
        connection = open_sqlite_connection(self.database_path)
        try:
            migrations.apply_migrations(connection)
            profile = InferenceProfileStore(connection).get(profile_id)
        finally:
            connection.close()
        if profile is None:
            raise InferenceProfileNotFoundError("inference profile was not found")
        if not profile.enabled:
            raise InferenceProfileDisabledError("inference profile is disabled")
        return profile

    def get_profile(self, profile_id: str) -> InferenceProfile:
        """Validate and return an enabled profile for service startup wiring."""

        return self._load_profile(profile_id)

    def _record_audit(
        self,
        *,
        task: ModelTask,
        profile: InferenceProfile,
        response: ModelResponse | None,
        status: str,
        error_code: str | None = None,
    ) -> None:
        connection = open_sqlite_connection(self.database_path)
        try:
            migrations.apply_migrations(connection)
            InferenceProfileStore(connection).record_egress_audit(
                memory_space_id=task.memory_space_id,
                profile_id=profile.profile_id,
                operation=task.operation,
                provider_kind=profile.provider_kind,
                model=profile.model,
                status=status,
                request_bytes=len(task.request.encoded_payload()),
                response_bytes=response.response_bytes if response is not None else 0,
                attempts=response.attempts if response is not None else 0,
                error_code=error_code,
            )
            connection.commit()
        finally:
            connection.close()

    def _run_task(
        self,
        task: ModelTask,
        profile_id: str,
        *,
        validator: Callable[[ModelResponse], object] | None = None,
    ) -> ModelResponse:
        profile = self._load_profile(profile_id)
        request_size = len(task.request.encoded_payload())
        input_limit = self.max_input_chars or profile.max_input_tokens * 4
        if request_size > input_limit:
            raise InferenceInputTooLargeError(
                "model request exceeds configured input limit"
            )

        secret = self.secret_store.get(profile.secret_ref)
        provider: InferenceProvider | None = None
        response: ModelResponse | None = None
        try:
            provider = self.provider_factory(profile, secret)
            response = provider.complete_json(task.request)
            if validator is not None:
                validator(response)
        except Exception as exc:
            error_code = getattr(exc, "code", "provider_error")
            if not isinstance(error_code, str):
                error_code = "provider_error"
            self._record_audit(
                task=task,
                profile=profile,
                response=response,
                status="error",
                error_code=error_code,
            )
            raise
        finally:
            if provider is not None:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()

        self._record_audit(
            task=task,
            profile=profile,
            response=response,
            status="success",
        )
        if response is None:
            raise InferenceBrokerError("provider returned no response")
        return response

    def execute_model_task(
        self, model_task: ModelTask, profile_id: str
    ) -> ModelResponse:
        return self._run_task(model_task, profile_id)

    def _build_hypothesis_task(
        self,
        normalized_round: NormalizedRound,
        profile: InferenceProfile,
    ) -> ModelTask:
        system_prompt = _hypothesis_prompt()
        user_prompt = json.dumps(
            {"normalized_round": _normalized_round_payload(normalized_round)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        request = ModelRequest.from_messages(
            model=profile.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=profile.max_output_tokens,
        )
        return ModelTask(
            memory_space_id=normalized_round.memory_space_id,
            operation="hypothesis",
            request=request,
        )

    @staticmethod
    def _validate_hypothesis_response(
        response: ModelResponse,
        normalized_round: NormalizedRound,
    ) -> tuple[HypothesisCandidate, ...]:
        try:
            parsed = HypothesisResponse.model_validate_json(response.content)
            candidates = parsed.to_candidates()
            for candidate in candidates:
                candidate.validate_source_events(normalized_round)
            return candidates
        except ValueError as exc:
            raise InferenceResponseValidationError(
                "provider response failed hypothesis schema validation"
            ) from exc

    def generate_hypotheses(
        self,
        normalized_round: NormalizedRound,
        profile_id: str,
    ) -> tuple[HypothesisCandidate, ...]:
        profile = self._load_profile(profile_id)
        task = self._build_hypothesis_task(normalized_round, profile)
        candidates: list[HypothesisCandidate] = []

        def validate(response: ModelResponse) -> None:
            candidates.extend(
                self._validate_hypothesis_response(response, normalized_round)
            )

        self._run_task(task, profile_id, validator=validate)
        return tuple(candidates)

    def generate_merge_proposal(
        self,
        *,
        memory_space_id: str,
        model_input: dict[str, object],
        profile_id: str,
    ) -> MergeProposal:
        """Generate one strict merge proposal through the configured provider."""

        profile = self._load_profile(profile_id)
        request = ModelRequest.from_messages(
            model=profile.model,
            system_prompt=_merge_prompt(),
            user_prompt=json.dumps(
                model_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            max_output_tokens=profile.max_output_tokens,
        )
        task = ModelTask(
            memory_space_id=memory_space_id,
            operation="merge",
            request=request,
        )
        proposal: MergeProposal | None = None

        def validate(response: ModelResponse) -> None:
            nonlocal proposal
            try:
                proposal = MergeProposal.model_validate_json(response.content)
            except ValueError as exc:
                raise InferenceResponseValidationError(
                    "provider response failed merge proposal schema validation"
                ) from exc

        self._run_task(task, profile_id, validator=validate)
        if proposal is None:
            raise InferenceResponseValidationError("provider returned no merge proposal")
        return proposal


__all__ = [
    "InferenceBroker",
    "InferenceBrokerError",
    "InferenceInputTooLargeError",
    "InferenceProfileDisabledError",
    "InferenceProfileNotFoundError",
    "InferenceResponseValidationError",
    "MergeProposal",
    "ModelTask",
]
