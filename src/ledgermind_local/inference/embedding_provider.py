"""Embedding provider contract over the Local technical backend."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..embedding_purpose import EmbeddingPurpose, validate_embedding_purpose
from .cancellation import CancellationToken
from .profiles import (
    EmbeddingProfileIdentity,
    EmbeddingProfileReadiness,
    InferenceProfile,
)
from .vectorizer import Vectorizer

VectorizerFactory = Callable[[], Vectorizer]


class EmbeddingBatch(BaseModel):
    """Typed embedding result with model identity metadata and no content coupling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vectors: tuple[tuple[float, ...], ...]
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    purpose: str = Field(min_length=1)


class EmbeddingError(RuntimeError):
    """Base embedding failure with a safe, structured code."""

    code = "embedding_error"


class EmbeddingRequestError(EmbeddingError):
    code = "embedding_request_error"


class EmbeddingBatchTooLargeError(EmbeddingError):
    code = "embedding_batch_too_large"


class EmbeddingTextTooLargeError(EmbeddingError):
    code = "embedding_text_too_large"


class EmbeddingNonFiniteError(EmbeddingError):
    code = "embedding_non_finite"


class EmbeddingDimensionMismatchError(EmbeddingError):
    code = "embedding_dimension_mismatch"


class EmbeddingModelError(EmbeddingError):
    code = "embedding_model_error"


class EmbeddingProvider:
    """Local embedding boundary wrapping the existing vectorizer, never copying it."""

    def __init__(
        self,
        *,
        vectorizer_factory: VectorizerFactory,
        max_texts: int = 64,
        max_text_chars: int = 8_000,
        identity_config: Mapping[str, object] | None = None,
    ) -> None:
        if max_texts < 1:
            raise ValueError("max_texts must be positive")
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")
        self._vectorizer_factory = vectorizer_factory
        self.max_texts = max_texts
        self.max_text_chars = max_text_chars
        self._identity_config = dict(identity_config or {})

    def describe_profile(self, profile: InferenceProfile) -> EmbeddingProfileIdentity:
        """Resolve opaque embedding identity without sending any text."""

        vectorizer = self._vectorizer_factory()
        try:
            return self._profile_identity(
                profile,
                vectorizer,
                self._vectorizer_dimension(vectorizer),
            )
        finally:
            close = getattr(vectorizer, "close", None)
            if callable(close):
                close()

    # Explicit alias for callers that name the contract rather than the action.
    embedding_profile_identity = describe_profile

    def readiness(self, profile: InferenceProfile) -> EmbeddingProfileReadiness:
        """Return secret-free profile readiness and identity metadata."""

        try:
            identity = self.describe_profile(profile)
        except Exception as exc:  # noqa: BLE001 - readiness must remain observable
            code = getattr(exc, "code", None)
            if not isinstance(code, str) or not code:
                code = "embedding_profile_unavailable"
            return EmbeddingProfileReadiness(
                profile_id=profile.profile_id,
                ready=False,
                error_code=code,
            )
        if identity.dimensions is None:
            return EmbeddingProfileReadiness(
                profile_id=profile.profile_id,
                ready=False,
                identity=identity,
                error_code="embedding_dimensions_unknown",
            )
        return EmbeddingProfileReadiness(
            profile_id=profile.profile_id,
            ready=True,
            identity=identity,
        )

    def embed(
        self,
        texts: Sequence[str],
        profile: InferenceProfile,
        purpose: EmbeddingPurpose,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> EmbeddingBatch:
        """Embed ``texts`` through the local backend and validate the batch."""
        token = cancellation_token or CancellationToken()
        token.raise_if_cancelled()
        if profile is None:
            raise EmbeddingRequestError("embedding profile is required")
        try:
            validate_embedding_purpose(purpose)
        except ValueError as exc:
            raise EmbeddingRequestError(str(exc)) from exc
        if not texts:
            raise EmbeddingRequestError("embedding batch must not be empty")
        if len(texts) > self.max_texts:
            raise EmbeddingBatchTooLargeError(
                f"embedding batch of {len(texts)} texts exceeds limit {self.max_texts}"
            )
        for text in texts:
            if len(text) > self.max_text_chars:
                raise EmbeddingTextTooLargeError(
                    f"embedding text of {len(text)} characters exceeds "
                    f"limit {self.max_text_chars}"
                )

        vectorizer = self._vectorizer_factory()
        try:
            token.raise_if_cancelled()
            raw_vectors = list(vectorizer.encode(texts))
            if len(raw_vectors) != len(texts):
                raise EmbeddingModelError("embedding backend returned a partial result")
            vectors = self._validated_vectors(vectorizer, raw_vectors)
            identity = self._profile_identity(
                profile,
                vectorizer,
                len(vectors[0]) if vectors else self._vectorizer_dimension(vectorizer),
            )
            return EmbeddingBatch(
                vectors=tuple(vectors),
                model=profile.model,
                model_version=identity.profile_fingerprint,
                dimensions=len(vectors[0]) if vectors else 0,
                purpose=purpose,
            )
        finally:
            close = getattr(vectorizer, "close", None)
            if callable(close):
                close()

    def _validated_vectors(
        self, vectorizer: Vectorizer, raw_vectors: Sequence[Sequence[float]]
    ) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        expected_dimension = self._vectorizer_dimension(vectorizer)
        for vector in raw_vectors:
            coerced = tuple(float(value) for value in vector)
            if not coerced:
                raise EmbeddingModelError("embedding backend returned an empty vector")
            for value in coerced:
                if not math.isfinite(value):
                    raise EmbeddingNonFiniteError(
                        "embedding vector contains non-finite values"
                    )
            if expected_dimension is not None and len(coerced) != expected_dimension:
                raise EmbeddingDimensionMismatchError(
                    f"embedding dimension {len(coerced)} does not match "
                    f"backend dimension {expected_dimension}"
                )
            vectors.append(coerced)
        dimension = len(vectors[0]) if vectors else 0
        if any(len(vector) != dimension for vector in vectors):
            raise EmbeddingDimensionMismatchError(
                "embedding batch has inconsistent vector dimensions"
            )
        return vectors

    @staticmethod
    def _vectorizer_dimension(vectorizer: Vectorizer) -> int | None:
        try:
            return int(vectorizer.dimension)
        except (RuntimeError, TypeError, ValueError):
            return None

    def _profile_identity(
        self,
        profile: InferenceProfile,
        vectorizer: Vectorizer,
        dimensions: int | None,
    ) -> EmbeddingProfileIdentity:
        try:
            model_version = vectorizer.fingerprint.strip()
        except (AttributeError, RuntimeError, TypeError):
            model_version = ""
        if not model_version or model_version == "unknown":
            raise EmbeddingModelError(
                "embedding backend did not expose a stable model fingerprint"
            )
        config: dict[str, object] = {
            "base_url": profile.base_url,
            "provider_kind": profile.provider_kind,
            "vectorizer": f"{type(vectorizer).__module__}.{type(vectorizer).__qualname__}",
            "max_input_tokens": profile.max_input_tokens,
            **self._identity_config,
        }
        return EmbeddingProfileIdentity(
            model_id=profile.model,
            model_version=model_version,
            dimensions=dimensions,
            config=config,
        )


__all__ = [
    "EmbeddingBatch",
    "EmbeddingBatchTooLargeError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingModelError",
    "EmbeddingNonFiniteError",
    "EmbeddingProfileIdentity",
    "EmbeddingProfileReadiness",
    "EmbeddingProvider",
    "EmbeddingPurpose",
    "EmbeddingRequestError",
    "EmbeddingTextTooLargeError",
    "VectorizerFactory",
]
