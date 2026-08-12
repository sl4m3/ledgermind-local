from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from ledgermind_local.inference.core_task_executor import (
    CoreTaskExecutor,
    GenericExecutionTask,
    ModelRequestSpec,
)
from ledgermind_local.inference.embedding_provider import EmbeddingProvider
from ledgermind_local.inference.profile_slots import (
    ProfileSlot,
    StoreBackedProfileResolver,
)
from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import (
    InferenceProfile,
    ProviderCapabilities,
)
from ledgermind_local.inference.provider_probe import ProviderProbe
from ledgermind_local.inference.providers.base import (
    ChatMessage,
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
    ProviderResponseError,
)
from ledgermind_local.inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
    build_payload_json_object,
    build_payload_json_schema,
    build_payload_prompt_only,
    build_payload_tool_call,
    decode_json_content,
)
from ledgermind_local.inference.secrets import SecretStore
from ledgermind_local.inference.structured_json_provider import StructuredJsonProvider
from ledgermind_local.inference.token_budget import InputBudgetExceededError
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.scheduler.core_execution_task_worker import (
    classify_execution_error,
)

CONTRACT = {
    "contract_name": "technical_result",
    "schema_version": 1,
    "json_schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    },
    "schema_digest": "sha256:" + "a" * 64,
}


class _FakeProvider:
    provider_kind = "openai_compatible"

    def __init__(self, *, failures: set[str] | None = None, content: str = '{"ok":true}') -> None:
        self.failures = failures or set()
        self.content = content
        self.requests: list[ModelRequest] = []

    def complete_json(self, request: ModelRequest, **_kwargs: object) -> ModelResponse:
        self.requests.append(request)
        if request.mode in self.failures:
            raise ProviderResponseError("mode rejected")
        return ModelResponse(
            content=self.content,
            raw_text=self.content,
            model=request.model,
            attempts=1,
            request_bytes=len(request.encoded_payload()),
            response_bytes=len(self.content.encode("utf-8")),
            status_code=200,
            mode=request.mode,
            output_contract=request.output_contract,
            tool_name=request.tool_name,
            metadata={"usage": {"prompt_tokens": 3}},
        )

    def close(self) -> None:
        return None


def _profile(*, max_input_tokens: int = 12_000, preference: str = "auto") -> InferenceProfile:
    return InferenceProfile(
        profile_id="profile",
        base_url="https://provider.example/v1",
        model="model",
        secret_ref="provider-secret",
        max_input_tokens=max_input_tokens,
        structured_output_preference=preference,
    )


def _store(profile: InferenceProfile | None = None) -> tuple[sqlite3.Connection, InferenceProfileStore]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    migrations.apply_migrations(connection)
    connection.execute(
        "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) "
        "VALUES ('space', 'tests', '2026-08-09T00:00:00Z', '2026-08-09T00:00:00Z')"
    )
    store = InferenceProfileStore(connection)
    store.upsert(profile or _profile())
    store.bind_slot("space", slot="operational", profile_id="profile")
    connection.commit()
    return connection, store


def _secret_store(tmp_path) -> SecretStore:
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("provider-secret", "TOP_SECRET")
    return secrets


def test_all_four_modes_have_separate_payload_shapes_and_capabilities() -> None:
    base = {
        "model": "model",
        "messages": (
            ChatMessage(role="system", content="technical role"),
            ChatMessage(role="user", content="return JSON"),
        ),
        "max_output_tokens": 42,
        "output_contract": CONTRACT,
        "tool_name": "submit_result",
        "token_parameter": "max_completion_tokens",
        "supports_system_role": False,
        "supports_seed": True,
        "seed": 7,
    }

    schema = build_payload_json_schema(ModelRequest(**base, mode="json_schema"))
    assert schema["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "technical_result",
            "strict": True,
            "schema": CONTRACT["json_schema"],
        },
    }
    assert "max_completion_tokens" in schema
    assert "max_tokens" not in schema
    assert schema["messages"][0]["role"] == "user"  # type: ignore[index]
    assert schema["seed"] == 7

    tool = build_payload_tool_call(ModelRequest(**base, mode="tool_call"))
    assert "response_format" not in tool
    assert tool["tools"][0]["function"]["name"] == "submit_result"  # type: ignore[index]

    object_payload = build_payload_json_object(ModelRequest(**base, mode="json_object"))
    assert object_payload["response_format"] == {"type": "json_object"}
    legacy = ModelRequest(
        model="model",
        messages=(ChatMessage(role="user", content="return"),),
        max_output_tokens=10,
        response_format={"type": "json_object"},
    )
    assert legacy.to_openai_payload()["response_format"] == {"type": "json_object"}

    prompt = build_payload_prompt_only(ModelRequest(**base, mode="prompt_only"))
    assert "response_format" not in prompt
    assert "Do not use Markdown" in prompt["messages"][-1]["content"]  # type: ignore[index]


def test_openai_provider_extracts_tool_arguments_and_fenced_json() -> None:
    responses = [
        {
            "model": "model",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "submit_result",
                                    "arguments": '{"ok":true}',
                                }
                            }
                        ]
                    }
                }
            ],
        },
        {
            "model": "model",
            "choices": [
                {"message": {"content": "```json\n{\"ok\":true}\n```"}}
            ],
        },
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    tool_response = provider.complete_json(
        ModelRequest(
            model="model",
            messages=(ChatMessage(role="user", content="return"),),
            max_output_tokens=10,
            mode="tool_call",
            output_contract=CONTRACT,
            tool_name="submit_result",
        )
    )
    fenced_response = provider.complete_json(
        ModelRequest(
            model="model",
            messages=(ChatMessage(role="user", content="return"),),
            max_output_tokens=10,
            mode="json_object",
        )
    )
    assert json.loads(tool_response.content) == {"ok": True}
    assert decode_json_content(fenced_response.content) == {"ok": True}
    provider.close()


def test_json_schema_is_rejected_before_http_when_contract_is_missing() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderConfigurationError):
        provider.complete_json(
            ModelRequest(
                model="model",
                messages=(ChatMessage(role="user", content="return"),),
                max_output_tokens=10,
                mode="json_schema",
            )
        )
    assert called is False
    provider.close()


def test_probe_auto_order_manual_override_and_capability_persistence(tmp_path) -> None:
    connection, store = _store()
    try:
        secret_store = _secret_store(tmp_path)
        auto_provider = _FakeProvider(failures={"json_schema"})
        auto_result = ProviderProbe(
            profile_resolver=StoreBackedProfileResolver(store),
            secret_store=secret_store,
            capability_store=store,
            provider_factory=lambda _profile, _secret: auto_provider,
        ).probe("space", ProfileSlot.OPERATIONAL)
        assert auto_result.attempted_modes == ("json_schema", "tool_call")
        assert auto_result.selected_mode == "tool_call"
        assert store.get_capabilities("profile") == auto_result.capabilities
        assert auto_result.capabilities.tool_call_supported is True

        store.upsert(_profile(preference="prompt_only"))
        manual_provider = _FakeProvider()
        manual_result = ProviderProbe(
            profile_resolver=StoreBackedProfileResolver(store),
            secret_store=secret_store,
            capability_store=store,
            provider_factory=lambda _profile, _secret: manual_provider,
        ).probe("space", "operational")
        assert manual_result.attempted_modes == ("prompt_only",)
        assert manual_result.selected_mode == "prompt_only"
    finally:
        connection.close()


def test_structured_provider_selects_capability_and_roundtrips_digest_and_fence(
    tmp_path,
) -> None:
    connection, store = _store()
    try:
        store.upsert_capabilities(
            ProviderCapabilities(
                profile_id="profile",
                structured_output_mode="tool_call",
                tool_call_supported=True,
                probe_contract_digest=CONTRACT["schema_digest"],
                probe_status="passed",
            )
        )
        fake = _FakeProvider(content="```json\n{\"ok\":true}\n```")
        result = StructuredJsonProvider(
            profile_resolver=StoreBackedProfileResolver(store),
            secret_store=_secret_store(tmp_path),
            capability_store=store,
            provider_factory=lambda _profile, _secret: fake,
        ).generate_json(
            memory_space_id="space",
            messages=(ChatMessage(role="user", content="return"),),
            max_output_tokens=20,
            profile_slot=ProfileSlot.OPERATIONAL,
            output_contract=CONTRACT,
        )
        assert fake.requests[0].mode == "tool_call"
        assert result.data == {"ok": True}
        assert result.structured_output_mode == "tool_call"
        assert result.contract_digest == CONTRACT["schema_digest"]
        assert result.raw_model_text.startswith("```")
        assert result.metadata["usage"] == {"prompt_tokens": 3}
    finally:
        connection.close()


def test_input_budget_rejection_happens_before_provider_factory(tmp_path) -> None:
    connection, store = _store(_profile(max_input_tokens=1))
    try:
        created = False

        def factory(_profile: InferenceProfile, _secret: str) -> _FakeProvider:
            nonlocal created
            created = True
            raise AssertionError("provider must not be constructed")

        provider = StructuredJsonProvider(
            profile_resolver=StoreBackedProfileResolver(store),
            secret_store=_secret_store(tmp_path),
            provider_factory=factory,
        )
        with pytest.raises(InputBudgetExceededError) as error:
            provider.generate_json(
                memory_space_id="space",
                messages=(ChatMessage(role="user", content="too much input"),),
                max_output_tokens=20,
                profile_slot=ProfileSlot.OPERATIONAL,
            )
        assert error.value.code == "input_budget_exceeded"
        assert created is False
    finally:
        connection.close()


def test_worker_classifies_budget_failure_as_safe_permanent_error() -> None:
    classification = classify_execution_error(InputBudgetExceededError(10, 1))

    assert classification.error_code == "input_budget_exceeded"
    assert classification.retryable is False
    assert classification.retry_after_seconds == 0


def test_executor_keeps_operation_opaque_and_returns_result_metadata(tmp_path) -> None:
    connection, store = _store()
    try:
        fake = _FakeProvider()
        executor = CoreTaskExecutor(
            json_provider=StructuredJsonProvider(
                profile_resolver=StoreBackedProfileResolver(store),
                secret_store=_secret_store(tmp_path),
                provider_factory=lambda _profile, _secret: fake,
            ),
            embedding_provider=EmbeddingProvider(vectorizer_factory=lambda: None),  # type: ignore[arg-type]
            profile_resolver=StoreBackedProfileResolver(store),
            generate_json_timeout_seconds=2,
            embed_texts_timeout_seconds=2,
        )
        task = GenericExecutionTask(
            task_id="task",
            task_kind="generate_json",
            operation="core_owned_facet_resolution_value_operation",
            operation_input={"facet": "opaque-to-local"},
            profile_slot=ProfileSlot.OPERATIONAL,
            model_request=ModelRequestSpec(
                messages=(ChatMessage(role="user", content="return"),),
                max_output_tokens=20,
                output_contract=CONTRACT,
            ),
            lease={"memory_space_id": "space"},
        )
        result = executor.execute(task)
        assert result.status == "completed"
        assert result.operation == task.operation
        assert result.operation_input == task.operation_input
        assert result.structured_output_mode == "json_object"
        assert result.contract_digest == CONTRACT["schema_digest"]
        assert result.raw_model_text == '{"ok":true}'
        assert fake.requests[0].metadata == {}
        assert "facet" not in json.dumps(fake.requests[0].model_dump())
    finally:
        connection.close()
