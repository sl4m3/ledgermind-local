from __future__ import annotations

import json

import httpx
import pytest

from ledgermind_local.installer.errors import ProviderProbeError
from ledgermind_local.installer.models import GenerationConfig
from ledgermind_local.installer.profiles.probes import probe_generation


def _config() -> GenerationConfig:
    return GenerationConfig(
        endpoint="https://provider.example/v1",
        token="secret",
        model="strict-model",
    )


def test_installer_generation_probe_uses_strict_schema_and_verifies_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "strict-model",
                "choices": [
                    {
                        "message": {
                            "content": '{"schema_version":1,"state":"ok","items":[1]}'
                        }
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe_generation(_config(), token="secret", client=client)

    assert seen["response_format"]["type"] == "json_schema"  # type: ignore[index]
    schema = seen["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert schema["additionalProperties"] is False  # type: ignore[index]
    assert schema["properties"]["state"]["enum"] == ["ok"]  # type: ignore[index]
    assert schema["properties"]["items"]["minItems"] == 1  # type: ignore[index]
    assert result["strict_structured_outputs"] is True
    assert result["schema_profile_version"] == "ledgermind-strict-schema"


def test_installer_generation_probe_rejects_parseable_but_wrong_shape() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderProbeError, match="strict probe schema"):
            probe_generation(_config(), token="secret", client=client)
