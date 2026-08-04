"""Tests for local service configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledgermind_local.config import LocalConfig


def test_config_version_required() -> None:
    with pytest.raises(ValidationError):
        LocalConfig.model_validate({"bind_host": "127.0.0.1"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LocalConfig.model_validate({"config_version": 1, "secret_api_key": "x"})


def test_remote_bind_requires_explicit_allow_remote_bind() -> None:
    with pytest.raises(ValidationError):
        LocalConfig.model_validate(
            {"config_version": 1, "bind_host": "0.0.0.0", "allow_remote_bind": False}
        )


def test_remote_bind_allowed_when_flag_enabled() -> None:
    config = LocalConfig.model_validate(
        {"config_version": 1, "bind_host": "0.0.0.0", "allow_remote_bind": True}
    )
    assert config.bind_host == "0.0.0.0"
    assert config.allow_remote_bind is True


def test_config_has_expected_6_2_shape() -> None:
    config = LocalConfig.model_validate(
        {
            "config_version": 1,
            "bind_host": "127.0.0.1",
            "bind_port": 8765,
        }
    )
    payload = config.to_json()
    assert '"config_version":1' in payload
    assert '"rounds_database_path":"rounds.db"' in payload
    assert '"log_level":"INFO"' in payload
    assert '"projection_poll_interval_seconds":1.0' in payload
    assert '"vector":{"enabled":false' in payload
    assert '"markdown_projection":{"enabled":false' in payload
    assert '"markdown_audit":{"enabled":false' in payload
    assert "markdown_audit_enabled" not in payload


def test_process_core_backend_is_the_only_default() -> None:
    config = LocalConfig.model_validate({"config_version": 1})

    assert config.core_backend == "process"


def test_python_core_backend_is_rejected_after_cutover() -> None:
    with pytest.raises(ValidationError):
        LocalConfig.model_validate({"config_version": 1, "core_backend": "python"})


def test_config_to_json_is_deterministic() -> None:
    config = LocalConfig.model_validate({"config_version": 1, "bind_port": 8765})
    first = config.to_json()
    second = config.to_json()
    assert first == second
    assert '"config_version":1' in first


def test_markdown_audit_disabled_by_default() -> None:
    config = LocalConfig.model_validate({"config_version": 1})
    assert config.markdown_audit_enabled is False


def test_markdown_audit_can_be_enabled() -> None:
    config = LocalConfig.model_validate(
        {"config_version": 1, "markdown_audit_enabled": True}
    )
    assert config.markdown_audit_enabled is True
    assert config.markdown_projection.enabled is True


def test_log_level_is_normalized_to_uppercase() -> None:
    config = LocalConfig.model_validate({"config_version": 1, "log_level": "info"})
    assert config.log_level == "INFO"
