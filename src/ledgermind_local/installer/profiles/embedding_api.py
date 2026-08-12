"""OpenAI-compatible embeddings client used by probes and local service."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

import httpx


class EmbeddingProviderError(RuntimeError):
    code = "embedding_provider_error"


class EmbeddingProviderAuthenticationError(EmbeddingProviderError):
    code = "authentication_failed"


class EmbeddingProviderTimeoutError(EmbeddingProviderError):
    code = "timeout"


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
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            raise ValueError("embedding batch must not be empty")
        if len(texts) > self.batch_size:
            raise EmbeddingProviderError(
                "embedding batch exceeds configured batch_size"
            )
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        try:
            response = self._client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingProviderTimeoutError("embedding request timed out") from exc
        except httpx.RequestError as exc:
            raise EmbeddingProviderError("embedding request failed") from exc
        if response.status_code in {401, 403}:
            raise EmbeddingProviderAuthenticationError(
                "embedding authentication failed"
            )
        if response.status_code >= 400:
            raise EmbeddingProviderError(
                f"embedding provider returned {response.status_code}"
            )
        try:
            envelope = response.json()
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
