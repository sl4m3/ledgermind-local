"""Small interactive wizard backed by the same InstallerConfig model."""

from __future__ import annotations

import getpass
from collections.abc import Callable
from typing import Literal, cast

from .errors import UserCancelledError
from .models import (
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
    LocalEmbeddingConfig,
    RuntimeConfig,
)


def _ask(
    prompt: str, *, input_fn: Callable[[str], str], default: str | None = None
) -> str:
    suffix = f" [{default}]" if default else ""
    value = input_fn(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def build_interactive_config(
    *,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
) -> InstallerConfig:
    try:
        target = _ask("Target", input_fn=input_fn, default="hermes")
        if target != "hermes":
            raise ValueError("only Hermes is supported in this release")
        semantic_language = _ask(
            "Semantic language (ru/en/es/pt/fr/de/uk)", input_fn=input_fn
        )
        if semantic_language not in {"ru", "en", "es", "pt", "fr", "de", "uk"}:
            raise ValueError("semantic language must be one of ru, en, es, pt, fr, de, uk")
        endpoint = _ask("Generation endpoint", input_fn=input_fn)
        token = secret_fn("Generation token: ").strip()
        model = _ask("Generation model", input_fn=input_fn)
        object_resolution_model = _ask(
            "Object Resolution model (separate profile)", input_fn=input_fn
        )
        embedding_mode = _ask(
            "Embeddings mode (api/local)", input_fn=input_fn, default="api"
        )
        if embedding_mode == "api":
            embedding_endpoint = _ask(
                "Embedding endpoint", input_fn=input_fn, default=endpoint
            )
            embedding_token = secret_fn("Embedding token: ").strip()
            embedding_model = _ask("Embedding model", input_fn=input_fn)
            dimensions = int(
                _ask("Embedding dimensions", input_fn=input_fn, default="1536")
            )
            embedding = EmbeddingConfig(
                mode="api",
                api=EmbeddingApiConfig(
                    endpoint=embedding_endpoint,
                    token=embedding_token,
                    model=embedding_model,
                    dimensions=dimensions,
                ),
            )
        elif embedding_mode == "local":
            catalog_id = _ask("Signed local model id", input_fn=input_fn)
            device = cast(
                Literal["auto", "cpu", "cuda", "rocm"],
                _ask(
                    "Device (auto/cpu/cuda/rocm)",
                    input_fn=input_fn,
                    default="auto",
                ),
            )
            embedding = EmbeddingConfig(
                mode="local",
                local=LocalEmbeddingConfig(catalog_id=catalog_id, device=device),
            )
        else:
            raise ValueError("embeddings mode must be api or local")
        idle = float(_ask("Idle shutdown seconds", input_fn=input_fn, default="60"))
        ttl = float(_ask("Lease TTL seconds", input_fn=input_fn, default="30"))
        heartbeat = float(_ask("Heartbeat seconds", input_fn=input_fn, default="10"))
        return InstallerConfig(
            target="hermes",
            semantic_language=semantic_language,
            generation=GenerationConfig(
                endpoint=endpoint,
                token=token,
                model=model,
                object_resolution_model=object_resolution_model,
            ),
            embedding=embedding,
            runtime=RuntimeConfig(
                idle_shutdown_seconds=idle,
                lease_ttl_seconds=ttl,
                heartbeat_seconds=heartbeat,
            ),
        )
    except (EOFError, KeyboardInterrupt) as exc:
        raise UserCancelledError("installation cancelled by user") from exc


__all__ = ["build_interactive_config"]
