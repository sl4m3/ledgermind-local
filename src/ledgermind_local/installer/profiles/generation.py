"""Generation profile construction for operational and background slots."""

from __future__ import annotations

from typing import Any

from ..models import GenerationConfig


def build_generation_profiles(
    config: GenerationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    operational = config.operational_model or config.model
    background = config.background_model or config.model
    common = {
        "endpoint": config.endpoint,
        "secret_ref": "generation/token",
        "timeout_seconds": config.timeout_seconds,
        "max_concurrency": config.max_concurrency,
        "structured_json_support": config.structured_json_support,
    }
    return (
        {
            "profile_id": "generation-operational",
            "slot": "operational",
            "model": operational,
            **common,
        },
        {
            "profile_id": "generation-background",
            "slot": "background",
            "model": background,
            **common,
        },
    )


__all__ = ["build_generation_profiles"]
