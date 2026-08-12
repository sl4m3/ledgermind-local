"""OpenAI-compatible embedding vectorizer for the Local task boundary."""

from __future__ import annotations

from collections.abc import Sequence

from ledgermind_local.installer.profiles.embedding_api import (
    OpenAICompatibleEmbeddingProvider,
)

from .vectorizer import Vectorizer


class OpenAIEmbeddingVectorizer(Vectorizer):
    """Adapt the installer API provider to Local's vectorizer protocol."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        model: str,
        dimensions: int,
        batch_size: int,
        timeout_seconds: float,
    ) -> None:
        self._provider = OpenAICompatibleEmbeddingProvider(
            endpoint=endpoint,
            token=token,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
        )
        self._model = model
        self._dimension = dimensions

    @property
    def fingerprint(self) -> str:
        return f"api:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self._provider.batch_size
        for start in range(0, len(texts), batch_size):
            batch = self._provider.embed(texts[start : start + batch_size])
            vectors.extend([list(vector) for vector in batch])
        return vectors

    def close(self) -> None:
        self._provider.close()


__all__ = ["OpenAIEmbeddingVectorizer"]
