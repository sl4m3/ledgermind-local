from __future__ import annotations

import json

import httpx
import pytest

from ledgermind_local.inference.providers.base import (
    ModelRequest,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeoutError,
    TransientProviderError,
)
from ledgermind_local.inference.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


def _request() -> ModelRequest:
    return ModelRequest.from_messages(
        model="test-model",
        system_prompt="Return JSON only.",
        user_prompt="Find hypotheses.",
        max_output_tokens=100,
    )


def test_openai_compatible_provider_posts_minimal_json_request_and_parses_response() -> (
    None
):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {"message": {"role": "assistant", "content": '{"hypotheses": []}'}}
                ],
            },
        )

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="TOP_SECRET",
        timeout_seconds=2,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = provider.complete_json(_request())

    assert seen["url"] == "https://provider.example/v1/chat/completions"
    assert seen["authorization"] == "Bearer TOP_SECRET"
    assert seen["payload"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Find hypotheses."},
        ],
        "max_tokens": 100,
        "response_format": {"type": "json_object"},
    }
    assert json.loads(response.content) == {"hypotheses": []}
    assert response.attempts == 1
    provider.close()


def test_openai_compatible_provider_rejects_auth_failure_without_secret_in_error() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "TOP_SECRET invalid"}})

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="TOP_SECRET",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderAuthenticationError) as error:
        provider.complete_json(_request())

    assert "TOP_SECRET" not in str(error.value)
    provider.close()


def test_openai_compatible_provider_retries_429_and_5xx(tmp_path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429 if calls == 1 else 503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_retries=2,
        retry_delay_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = provider.complete_json(_request())

    assert calls == 3
    assert response.attempts == 3
    provider.close()


def test_openai_compatible_provider_surfaces_transient_failure_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_retries=1,
        retry_delay_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(TransientProviderError, match="temporary"):
        provider.complete_json(_request())
    provider.close()


def test_openai_compatible_provider_handles_timeout_and_invalid_response() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("network timeout", request=request)

    timeout_provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(timeout_handler)),
    )
    with pytest.raises(ProviderTimeoutError):
        timeout_provider.complete_json(_request())
    timeout_provider.close()

    def invalid_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not-json"}}]}
        )

    invalid_provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(invalid_handler)),
    )
    with pytest.raises(ProviderResponseError, match="JSON"):
        invalid_provider.complete_json(_request())
    invalid_provider.close()


def test_openai_compatible_provider_limits_response_size() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 101)

    provider = OpenAICompatibleProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        max_response_bytes=100,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderResponseError, match="size"):
        provider.complete_json(_request())
    provider.close()
