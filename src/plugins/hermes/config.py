"""Configuration loading for the local Hermes plugin."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUIRED_STRINGS = (
    "source_instance_id",
    "memory_space_id",
)


@dataclass(frozen=True, slots=True)
class PluginConfig:
    config_version: int
    source_instance_id: str
    service_url: str
    token_file: str
    profile_name: str
    memory_space_id: str
    state_db_path: str
    extraction_prompt_version: int
    extraction_schema_version: int
    pre_llm_timeout_seconds: float
    delivery_timeout_seconds: float
    max_context_items: int

    @property
    def expanded_token_file(self) -> Path:
        return Path(self.token_file).expanduser()

    @property
    def expanded_state_db_path(self) -> Path:
        return Path(self.state_db_path).expanduser()


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_str(value: Any, *, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _coerce_str_required(payload: dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    value = "" if raw is None else str(raw).strip()
    if not value:
        raise ValueError(f"configuration field is required: {key}")
    return value


def load_config(path: str | Path) -> PluginConfig:
    config_path = Path(path).expanduser()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("plugin configuration must be a JSON object")

    for key in _REQUIRED_STRINGS:
        _coerce_str_required(payload, key)

    return PluginConfig(
        config_version=_coerce_int(payload.get("config_version"), default=1),
        source_instance_id=_coerce_str_required(payload, "source_instance_id"),
        service_url=_coerce_str(
            payload.get("service_url"),
            default="http://127.0.0.1:8000",
        ),
        token_file=_coerce_str(
            payload.get("token_file"),
            default="~/.ledgermind/local/server.token",
        ),
        profile_name=_coerce_str(payload.get("profile_name"), default="default"),
        memory_space_id=_coerce_str_required(payload, "memory_space_id"),
        state_db_path=_coerce_str(
            payload.get("state_db_path"),
            default="~/.hermes/state.db",
        ),
        extraction_prompt_version=_coerce_int(
            payload.get("extraction_prompt_version"),
            default=1,
        ),
        extraction_schema_version=_coerce_int(
            payload.get("extraction_schema_version"),
            default=1,
        ),
        pre_llm_timeout_seconds=_coerce_float(
            payload.get("pre_llm_timeout_seconds"),
            default=1.0,
        ),
        delivery_timeout_seconds=_coerce_float(
            payload.get("delivery_timeout_seconds"),
            default=5.0,
        ),
        max_context_items=_coerce_int(payload.get("max_context_items"), default=5),
    )
