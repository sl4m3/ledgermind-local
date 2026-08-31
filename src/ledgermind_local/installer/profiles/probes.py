"""Real provider probes performed by install, configure, and doctor."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from ..errors import ProviderProbeError
from ...inference.strict import (
    STRICT_SCHEMA_PROFILE_VERSION,
    canonical_digest,
    validate_strict_schema_profile,
)
from ...inference.providers.openai_compatible import provider_request_headers
from ..models import EmbeddingApiConfig, GenerationConfig
from .embedding_api import OpenAICompatibleEmbeddingProvider


def _safe_status(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise ProviderProbeError("provider authentication failed")
    if response.status_code == 404:
        raise ProviderProbeError("provider endpoint or model was not found")
    if response.status_code == 429:
        raise ProviderProbeError("provider rate limit was reached")
    if response.status_code >= 400:
        raise ProviderProbeError(f"provider returned HTTP {response.status_code}")


_STRICT_PROBE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "state": {"type": "string", "enum": ["ok"]},
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "integer"},
        },
    },
    "required": ["schema_version", "state", "items"],
}


def _strict_probe_output(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        set(payload) == {"schema_version", "state", "items"}
        and payload.get("schema_version") == 1
        and payload.get("state") == "ok"
        and isinstance(payload.get("items"), list)
        and 1 <= len(payload["items"]) <= 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in payload["items"])
    )


def probe_generation(
    config: GenerationConfig, *, token: str, client: httpx.Client | None = None
) -> dict[str, Any]:
    if not token:
        raise ProviderProbeError("generation token is empty")
    owns_client = client is None
    active = client or httpx.Client(timeout=config.timeout_seconds)
    try:
        endpoint = config.endpoint.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        response = active.post(
            endpoint,
            headers=provider_request_headers(endpoint, token),
            json={
                "model": config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return exactly one JSON object matching the strict capability probe schema.",
                    },
                    {
                        "role": "user",
                        "content": 'Return {"schema_version":1,"state":"ok","items":[1]}.',
                    },
                ],
                "max_tokens": 32,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ledgermind_strict_probe",
                        "strict": True,
                        "schema": _STRICT_PROBE_SCHEMA,
                    },
                },
            },
            timeout=config.timeout_seconds,
        )
        _safe_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderProbeError("generation response is not JSON") from exc
        content = (
            payload.get("choices", [{}])[0].get("message", {}).get("content")
            if isinstance(payload, dict)
            and isinstance(payload.get("choices"), list)
            and payload["choices"]
            else None
        )
        if not isinstance(content, str):
            raise ProviderProbeError("generation response has no message content")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderProbeError(
                "generation provider did not return structured JSON"
            ) from exc
        if not _strict_probe_output(decoded):
            raise ProviderProbeError(
                "generation provider did not satisfy the strict probe schema"
            )
        return {
            "endpoint": config.endpoint,
            "model": payload.get("model", config.model)
            if isinstance(payload, dict)
            else config.model,
            "request_id": response.headers.get("x-request-id")
            or response.headers.get("request-id"),
            "status_code": response.status_code,
            "structured_json": True,
            "strict_structured_outputs": True,
            "schema_profile_version": STRICT_SCHEMA_PROFILE_VERSION,
            "probe_contract_digest": canonical_digest(
                validate_strict_schema_profile(_STRICT_PROBE_SCHEMA)
            ),
        }
    except httpx.TimeoutException as exc:
        raise ProviderProbeError("generation provider timed out") from exc
    except httpx.RequestError as exc:
        raise ProviderProbeError("generation provider request failed") from exc
    finally:
        if owns_client:
            active.close()


def probe_embedding_api(
    config: EmbeddingApiConfig,
    *,
    token: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    provider = OpenAICompatibleEmbeddingProvider(
        endpoint=config.endpoint,
        token=token,
        model=config.model,
        dimensions=config.dimensions,
        batch_size=config.batch_size,
        timeout_seconds=config.timeout_seconds,
        client=client,
    )
    try:
        one = provider.embed(("LedgerMind embedding probe",))
        batch = provider.embed(("LedgerMind embedding probe", "LedgerMind batch probe"))
        if len(one) != 1 or len(batch) != 2:
            raise ProviderProbeError("embedding provider returned an invalid batch")
        for vector in (*one, *batch):
            if len(vector) != config.dimensions or not all(
                math.isfinite(value) for value in vector
            ):
                raise ProviderProbeError(
                    "embedding provider returned invalid dimensions or floats"
                )
        return {
            "endpoint": config.endpoint,
            "model": config.model,
            "dimensions": config.dimensions,
            "batch_size": config.batch_size,
            "request_count": 2,
            "status": "passed",
        }
    except ProviderProbeError:
        raise
    except Exception as exc:
        raise ProviderProbeError("embedding provider probe failed") from exc
    finally:
        provider.close()


__all__ = ["probe_embedding_api", "probe_generation"]
