"""OpenAI-compatible embeddings client used by probes and local service."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

import httpx

from ...inference.provider_telemetry import record_http_attempt
from ...inference.providers.openai_compatible import provider_request_headers


class EmbeddingProviderError(RuntimeError):
    code = "embedding_provider_error"


class EmbeddingProviderAuthenticationError(EmbeddingProviderError):
    code = "authentication_failed"


class EmbeddingProviderTimeoutError(EmbeddingProviderError):
    code = "timeout"


EmbeddingRole: TypeAlias = Literal["query", "passage"]


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        model: str,
        dimensions: int,
        batch_size: int = 32,
        max_concurrency: int = 1,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
        profile_fingerprint: str | None = None,
    ) -> None:
        normalized = endpoint.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("embedding endpoint must be an absolute http(s) URL")
        if not token:
            raise ValueError("embedding token must not be empty")
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.endpoint = (
            normalized
            if parsed.path.endswith("/embeddings")
            else f"{normalized}/embeddings"
        )
        self.token = token
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(int(batch_size), 1)
        self.max_concurrency = max(int(max_concurrency), 1)
        self.timeout = float(timeout_seconds)
        self._profile_fingerprint = profile_fingerprint
        self._operation = "embedding"
        self._operation_item_counts: dict[str, int] | None = None
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None

    def set_telemetry_context(
        self,
        *,
        operation: str,
        profile_fingerprint: str,
        operation_item_counts: dict[str, int] | None = None,
    ) -> None:
        self._operation = str(operation)
        self._profile_fingerprint = str(profile_fingerprint)
        self._operation_item_counts = (
            {str(key): int(value) for key, value in operation_item_counts.items()}
            if operation_item_counts is not None
            else None
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(
        self,
        texts: Sequence[str],
        *,
        role: EmbeddingRole | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            raise ValueError("embedding batch must not be empty")
        if len(texts) > self.batch_size:
            raise EmbeddingProviderError(
                "embedding batch exceeds configured batch_size"
            )
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        # Object retrieval is a dual-encoder contract.  Do not emulate a
        # missing role by sending a plain embedding request: callers that
        # need query/passage semantics must fail before reaching this layer.
        if role is not None:
            payload["input_type"] = role
        started = time.perf_counter()
        try:
            response = self._client.post(
                self.endpoint,
                headers=provider_request_headers(self.endpoint, self.token),
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            record_http_attempt(
                kind="embedding",
                operation=self._operation,
                provider_profile_fingerprint=self._profile_fingerprint,
                transport="openai_compatible",
                model=self.model,
                duration_ms=(time.perf_counter() - started) * 1000,
                status="timeout",
                batch_item_count=len(texts),
                operation_item_counts=self._operation_item_counts,
            )
            raise EmbeddingProviderTimeoutError("embedding request timed out") from exc
        except httpx.RequestError as exc:
            record_http_attempt(
                kind="embedding",
                operation=self._operation,
                provider_profile_fingerprint=self._profile_fingerprint,
                transport="openai_compatible",
                model=self.model,
                duration_ms=(time.perf_counter() - started) * 1000,
                status="transport_error",
                batch_item_count=len(texts),
                operation_item_counts=self._operation_item_counts,
            )
            raise EmbeddingProviderError("embedding request failed") from exc
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "request-id"
        ) or f"local-{uuid.uuid4().hex}"
        response_payload: object | None = None
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens")) if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        record_http_attempt(
            kind="embedding",
            operation=self._operation,
            provider_profile_fingerprint=self._profile_fingerprint,
            transport="openai_compatible",
            model=self.model,
            duration_ms=(time.perf_counter() - started) * 1000,
            status="completed" if response.status_code < 400 else "failed",
            request_id=request_id,
            http_status=response.status_code,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            usage_unknown=not isinstance(usage, dict),
            batch_item_count=len(texts),
            operation_item_counts=self._operation_item_counts,
        )
        if response.status_code in {401, 403}:
            raise EmbeddingProviderAuthenticationError(
                "embedding authentication failed"
            )
        if response.status_code >= 400:
            raise EmbeddingProviderError(
                f"embedding provider returned {response.status_code}"
            )
        try:
            envelope = response_payload
        except ValueError as exc:
            raise EmbeddingProviderError("embedding response is not JSON") from exc
        data = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingProviderError(
                "embedding response has an invalid data length"
            )
        response_model = envelope.get("model") if isinstance(envelope, dict) else None
        if response_model is not None and response_model != self.model:
            raise EmbeddingProviderError(
                "embedding response model does not match configuration"
            )
        ordered = sorted(
            data,
            key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
        )
        vectors: list[tuple[float, ...]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                raise EmbeddingProviderError("embedding response contains no vector")
            values = tuple(float(value) for value in vector)
            if len(values) != self.dimensions:
                raise EmbeddingProviderError(
                    "embedding dimensions do not match configuration"
                )
            if not all(math.isfinite(value) for value in values):
                raise EmbeddingProviderError(
                    "embedding vector contains non-finite values"
                )
            vectors.append(values)
        return tuple(vectors)


__all__ = [
    "EmbeddingProviderAuthenticationError",
    "EmbeddingProviderError",
    "EmbeddingProviderTimeoutError",
    "OpenAICompatibleEmbeddingProvider",
]
