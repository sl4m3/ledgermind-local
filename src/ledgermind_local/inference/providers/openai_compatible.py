"""OpenAI-compatible JSON completion provider."""

from __future__ import annotations

import json
import time
from urllib.parse import urlparse

import httpx

from .base import (
    InferenceProvider,
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTransportError,
    TransientProviderError,
)


class OpenAICompatibleProvider(InferenceProvider):
    """Synchronous provider for OpenAI-compatible ``/chat/completions`` APIs."""

    provider_kind = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_response_bytes: int = 2_000_000,
        retry_delay_seconds: float = 0.25,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderConfigurationError("base_url must be an absolute http(s) URL")
        if not api_key:
            raise ProviderConfigurationError("provider secret is not configured")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise ProviderConfigurationError(
                "timeout_seconds is outside the supported range"
            )
        if max_retries < 0 or max_retries > 5:
            raise ProviderConfigurationError(
                "max_retries is outside the supported range"
            )
        if max_response_bytes <= 0:
            raise ProviderConfigurationError("max_response_bytes must be positive")
        if retry_delay_seconds < 0 or retry_delay_seconds > 30:
            raise ProviderConfigurationError(
                "retry_delay_seconds is outside the supported range"
            )

        self.base_url = normalized_url
        self._api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_response_bytes = int(max_response_bytes)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self._client = client or httpx.Client(timeout=self.timeout_seconds)
        self._owns_client = client is None

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleProvider("
            f"base_url={self.base_url!r}, max_retries={self.max_retries}, "
            f"max_response_bytes={self.max_response_bytes})"
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, request: ModelRequest) -> ModelResponse:
        payload = request.to_openai_payload()
        request_bytes = len(request.encoded_payload())
        attempts = 0
        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise ProviderTimeoutError("provider request timed out") from exc
            except httpx.RequestError as exc:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise ProviderTransportError("provider transport failed") from exc

            if response.status_code in {401, 403}:
                raise ProviderAuthenticationError("provider authentication failed")
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise TransientProviderError(
                    f"provider returned temporary HTTP status {response.status_code}"
                )
            if response.status_code >= 400:
                raise ProviderResponseError(
                    f"provider returned HTTP status {response.status_code}"
                )

            response_bytes = len(response.content)
            if response_bytes > self.max_response_bytes:
                raise ProviderResponseError("provider response exceeds size limit")
            return self._parse_response(
                response,
                request=request,
                attempts=attempts,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
            )

        raise ProviderTransportError("provider request did not complete")

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_delay_seconds:
            time.sleep(min(self.retry_delay_seconds * attempt, 30.0))

    @staticmethod
    def _parse_response(
        response: httpx.Response,
        *,
        request: ModelRequest,
        attempts: int,
        request_bytes: int,
        response_bytes: int,
    ) -> ModelResponse:
        try:
            envelope = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                "provider returned invalid response JSON"
            ) from exc
        if not isinstance(envelope, dict):
            raise ProviderResponseError("provider response envelope is invalid")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response has no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderResponseError("provider response choice is invalid")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError("provider response message is invalid")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("provider response content is missing")
        try:
            decoded = json.loads(content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                "provider returned invalid JSON content"
            ) from exc
        if not isinstance(decoded, (dict, list)):
            raise ProviderResponseError(
                "provider JSON content must be an object or array"
            )
        response_model = envelope.get("model", request.model)
        if not isinstance(response_model, str) or not response_model:
            response_model = request.model
        return ModelResponse(
            content=content,
            model=response_model,
            attempts=attempts,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            status_code=response.status_code,
        )

    def complete_json(self, request: ModelRequest) -> ModelResponse:
        return self._request(request)


__all__ = ["OpenAICompatibleProvider"]
