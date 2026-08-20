from __future__ import annotations

import json

import httpx
import pytest

from ledgermind_local.inference.gguf_vectorizer import GGUFVectorizer
from ledgermind_local.inference.openai_vectorizer import OpenAIEmbeddingVectorizer
from ledgermind_local.installer.profiles.embedding_api import (
    OpenAICompatibleEmbeddingProvider,
)
from ledgermind_local.inference.vectorizer import VectorizerRoleError


def _provider(seen: list[dict[str, object]]) -> OpenAICompatibleEmbeddingProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "embed-model",
                "data": [
                    {"index": index, "embedding": [float(index + 1), 0.0]}
                    for index in range(len(json.loads(request.content)["input"]))
                ],
            },
        )

    return OpenAICompatibleEmbeddingProvider(
        endpoint="https://provider.example/v1",
        token="secret",
        model="embed-model",
        dimensions=2,
        batch_size=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("role", ["query", "passage"])
def test_openai_embedding_wire_preserves_role(role: str) -> None:
    seen: list[dict[str, object]] = []
    provider = _provider(seen)
    try:
        provider.embed(("one", "two"), role=role)  # type: ignore[arg-type]
    finally:
        provider.close()

    assert seen == [
        {
            "model": "embed-model",
            "input": ["one", "two"],
            "input_type": role,
        }
    ]


def test_openai_vectorizer_keeps_one_role_for_every_batch() -> None:
    seen: list[dict[str, object]] = []
    provider = _provider(seen)
    vectorizer = OpenAIEmbeddingVectorizer(
        endpoint="https://provider.example/v1",
        token="secret",
        model="embed-model",
        dimensions=2,
        batch_size=2,
        timeout_seconds=2,
    )
    # Replace the concrete adapter's HTTP client with the deterministic mock
    # provider used above; the assertion is about the adapter wire boundary.
    vectorizer._provider.close()  # noqa: SLF001 - test-only transport setup
    vectorizer._provider = provider  # noqa: SLF001
    try:
        vectorizer.encode(("one", "two", "three"), role="query")
    finally:
        vectorizer.close()

    assert [payload["input_type"] for payload in seen] == ["query", "query"]
    assert [payload["input"] for payload in seen] == [["one", "two"], ["three"]]


def test_gguf_rejects_required_role_instead_of_using_plain_embeddings() -> None:
    vectorizer = GGUFVectorizer(
        model_path="/tmp/model.gguf",
        model_builder=lambda *_: pytest.fail("GGUF model must not be opened"),
    )

    with pytest.raises(VectorizerRoleError, match="does not support"):
        vectorizer.encode(("query",), role="query")
