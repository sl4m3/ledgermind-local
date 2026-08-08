from __future__ import annotations

from collections.abc import Sequence

import pytest

from ledgermind_local.inference.cancellation import CancellationToken
from ledgermind_local.inference.embedding_provider import (
    EmbeddingBatchTooLargeError,
    EmbeddingDimensionMismatchError,
    EmbeddingModelError,
    EmbeddingNonFiniteError,
    EmbeddingProvider,
    EmbeddingRequestError,
    EmbeddingTextTooLargeError,
)
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.inference.providers.base import ProviderCancelledError


class _FakeVectorizer:
    def __init__(
        self,
        *,
        dimension: int,
        fingerprint: str,
        vectors: list[list[float]] | None = None,
        partial: bool = False,
        closed: list[bool] | None = None,
    ) -> None:
        self._dimension = dimension
        self._fingerprint = fingerprint
        self._vectors = vectors
        self._partial = partial
        self._closed = closed

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._partial:
            return [[0.0] * self._dimension] * max(0, len(texts) - 1)
        if self._vectors is not None:
            return self._vectors
        return [[float(len(text))] * self._dimension for text in texts]

    def close(self) -> None:
        if self._closed is not None:
            self._closed.append(True)


def _profile() -> InferenceProfile:
    return InferenceProfile(
        profile_id="embedding-default",
        base_url="https://provider.example/v1",
        model="embed-model",
        secret_ref="embed-secret",
    )


def _provider(
    vectorizer: _FakeVectorizer,
    *,
    max_texts: int = 64,
    max_text_chars: int = 8_000,
) -> EmbeddingProvider:
    return EmbeddingProvider(
        vectorizer_factory=lambda: vectorizer, max_texts=max_texts,
        max_text_chars=max_text_chars,
    )


def test_embed_returns_batch_with_model_metadata() -> None:
    vectorizer = _FakeVectorizer(dimension=3, fingerprint="sha256:abc")
    batch = _provider(vectorizer).embed(
        ["first text", "second"], _profile(), "knowledge"
    )

    assert len(batch.vectors) == 2
    assert all(len(vector) == 3 for vector in batch.vectors)
    assert batch.dimensions == 3
    assert batch.model == "embed-model"
    assert batch.model_version == "sha256:abc"
    assert batch.purpose == "knowledge"


def test_embed_closes_vectorizer_after_use() -> None:
    closed: list[bool] = []
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1", closed=closed)

    _provider(vectorizer).embed(["text"], _profile(), "knowledge")

    assert closed == [True]


def test_embed_rejects_empty_batch() -> None:
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1")

    with pytest.raises(EmbeddingRequestError):
        _provider(vectorizer).embed([], _profile(), "knowledge")


def test_embed_rejects_batch_over_limit() -> None:
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1")

    with pytest.raises(EmbeddingBatchTooLargeError):
        _provider(vectorizer, max_texts=2).embed(
            ["a", "b", "c"], _profile(), "knowledge"
        )


def test_embed_rejects_oversized_text() -> None:
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1")

    with pytest.raises(EmbeddingTextTooLargeError):
        _provider(vectorizer, max_text_chars=4).embed(
            ["way too long"], _profile(), "knowledge"
        )


def test_embed_rejects_non_finite_vectors() -> None:
    vectorizer = _FakeVectorizer(
        dimension=2, fingerprint="m1", vectors=[[1.0, float("nan")], [2.0, 2.0]]
    )

    with pytest.raises(EmbeddingNonFiniteError) as error:
        _provider(vectorizer).embed(["a", "b"], _profile(), "knowledge")

    assert error.value.code == "embedding_non_finite"


def test_embed_rejects_vector_dimension_mismatch_with_backend() -> None:
    vectorizer = _FakeVectorizer(
        dimension=3, fingerprint="m1", vectors=[[1.0, 2.0], [3.0, 4.0]]
    )

    with pytest.raises(EmbeddingDimensionMismatchError) as error:
        _provider(vectorizer).embed(["a", "b"], _profile(), "knowledge")

    assert error.value.code == "embedding_dimension_mismatch"


def test_embed_rejects_inconsistent_batch_dimensions() -> None:
    vectorizer = _FakeVectorizer(
        dimension=None,  # dimension unknown, backend returns mixed widths
        fingerprint="m1",
        vectors=[[1.0, 2.0], [1.0, 2.0, 3.0]],
    )

    with pytest.raises(EmbeddingDimensionMismatchError):
        _provider(vectorizer).embed(["a", "b"], _profile(), "knowledge")


def test_embed_rejects_partial_backend_result() -> None:
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1", partial=True)

    with pytest.raises(EmbeddingModelError) as error:
        _provider(vectorizer).embed(["a", "b", "c"], _profile(), "knowledge")

    assert error.value.code == "embedding_model_error"


def test_embed_rejects_pre_cancelled_token() -> None:
    vectorizer = _FakeVectorizer(dimension=2, fingerprint="m1")
    token = CancellationToken()
    token.cancel()

    with pytest.raises(ProviderCancelledError):
        _provider(vectorizer).embed(
            ["text"], _profile(), "knowledge", cancellation_token=token
        )


def test_embed_validates_constructor_limits() -> None:
    with pytest.raises(ValueError):
        _provider(_FakeVectorizer(dimension=2, fingerprint="m1"), max_texts=0)
    with pytest.raises(ValueError):
        _provider(_FakeVectorizer(dimension=2, fingerprint="m1"), max_text_chars=0)
