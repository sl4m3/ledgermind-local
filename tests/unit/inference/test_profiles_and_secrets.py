from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.inference.secrets import SecretNotFoundError, SecretStore


def test_inference_profile_forbids_provider_secret_and_unknown_fields() -> None:
    profile = InferenceProfile(
        profile_id="operational-default",
        provider_kind="openai_compatible",
        base_url="https://provider.example/v1",
        model="test-model",
        secret_ref="provider-main",
        timeout_seconds=12,
        max_retries=2,
        max_input_tokens=4_000,
        max_output_tokens=800,
        enabled=True,
    )

    assert profile.base_url == "https://provider.example/v1"
    assert "api_key" not in profile.model_dump()
    with pytest.raises(ValueError):
        InferenceProfile.model_validate({**profile.model_dump(), "api_key": "secret"})


def test_inference_profile_rejects_invalid_provider_and_limits() -> None:
    with pytest.raises(ValueError, match="provider_kind"):
        InferenceProfile(
            profile_id="p",
            provider_kind="anthropic",
            base_url="https://provider.example/v1",
            model="model",
            secret_ref="ref",
        )

    with pytest.raises(ValueError, match="max_output_tokens"):
        InferenceProfile(
            profile_id="p",
            provider_kind="openai_compatible",
            base_url="https://provider.example/v1",
            model="model",
            secret_ref="ref",
            max_output_tokens=0,
        )


def test_secret_store_is_private_atomic_and_does_not_expose_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "local" / "secrets.json"
    store = SecretStore(path)
    store.put("provider-main", "TOP_SECRET")

    assert store.get("provider-main") == "TOP_SECRET"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "TOP_SECRET" not in repr(store)
    assert "TOP_SECRET" in path.read_text(encoding="utf-8")
    assert (
        json.loads(path.read_text(encoding="utf-8"))["secrets"]["provider-main"]
        == "TOP_SECRET"
    )

    store.delete("provider-main")
    with pytest.raises(SecretNotFoundError, match="provider-main"):
        store.get("provider-main")


def test_secret_store_errors_do_not_include_secret_value(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    path.write_text('{"secrets": ["invalid"]}', encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        SecretStore(path).get("provider-main")

    assert "TOP_SECRET" not in str(error.value)
