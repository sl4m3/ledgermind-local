"""Configure provider profiles without installing a target automatically."""

from __future__ import annotations

from typing import Any

from ..config_writer import resolve_provider_tokens, write_installer_config
from ..models import InstallerConfig
from ..paths import InstallerPaths
from ..profiles.probes import probe_embedding_api, probe_generation


def configure(
    *,
    config: InstallerConfig,
    paths: InstallerPaths,
    validate_providers: bool = False,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    if validate_providers:
        generation_token, embedding_token = resolve_provider_tokens(
            config,
            paths,
            generation_stdin=generation_stdin,
            embedding_stdin=embedding_stdin,
        )
        probes["generation"] = probe_generation(
            config.generation, token=generation_token
        )
        if config.embedding.mode == "api" and config.embedding.api is not None:
            assert embedding_token is not None
            probes["embedding"] = probe_embedding_api(
                config.embedding.api, token=embedding_token
            )
    if dry_run:
        return {"status": "dry_run", "probes": probes}
    metadata = write_installer_config(
        config,
        paths,
        generation_stdin=generation_stdin,
        embedding_stdin=embedding_stdin,
    )
    return {"status": "success", "config": metadata, "probes": probes}


__all__ = ["configure"]
