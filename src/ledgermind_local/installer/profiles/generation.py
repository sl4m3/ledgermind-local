"""Generation profile construction for all production generation slots."""

from __future__ import annotations

from typing import Any

from ..models import GenerationConfig
from ..secret_refs import GENERATION_SECRET_REF


def build_generation_profiles(
    config: GenerationConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    operational = config.operational_model or config.model
    object_resolution = config.object_resolution_model
    if not object_resolution:
        raise ValueError(
            "generation.object_resolution_model is required; refusing operational fallback"
        )
    background = config.background_model or config.model
    common = {
        "endpoint": config.endpoint,
        "secret_ref": GENERATION_SECRET_REF,
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
            "profile_id": "generation-object-resolution",
            "slot": "object_resolution",
            "model": object_resolution,
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
