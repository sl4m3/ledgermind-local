"""Configure provider profiles without installing a target automatically."""

from __future__ import annotations

from typing import Any

from ledgermind_local.runtime.supervisor import RuntimeSupervisor

from ..config_writer import (
    persist_generation_probe,
    resolve_provider_tokens,
    write_installer_config,
    write_local_config,
    write_local_profiles,
)
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
    # A provider reconfiguration is explicit, but it must not race a live
    # worker that still holds the previous profile cache.
    RuntimeSupervisor(paths).stop(force=True)
    metadata = write_installer_config(
        config,
        paths,
        generation_stdin=generation_stdin,
        embedding_stdin=embedding_stdin,
    )
    metadata["local_config"] = str(write_local_config(config, paths))
    metadata["local_profiles"] = write_local_profiles(config, paths)
    if validate_providers:
        metadata["generation_capabilities"] = persist_generation_probe(
            config, paths, probes["generation"]
        )
    generation_status = probes.get("generation", {}).get(
        "status", "configured-not-probed"
    )
    generation_route = " -> ".join(
        item
        for item in (config.generation.route, *config.generation.fallback_routes)
        if item
    )
    generation_readiness = f"{generation_status}; model={config.generation.model}" + (
        f"; route={generation_route}" if generation_route else ""
    )
    if config.embedding.mode == "api":
        assert config.embedding.api is not None
        embedding_status = probes.get("embedding", {}).get(
            "status", "configured-not-probed"
        )
        embedding_readiness = (
            f"{embedding_status}; model={config.embedding.api.model}; "
            f"dimensions={config.embedding.api.dimensions}"
        )
    else:
        assert config.embedding.local is not None
        embedding_readiness = (
            f"local-configured; model={config.embedding.local.catalog_id}"
        )
    return {
        "status": "success",
        "config": metadata,
        "probes": probes,
        "readiness": {
            "platform": "linux",
            "core": "restart-on-next-agent-use",
            "generation": generation_readiness,
            "embeddings": embedding_readiness,
            "agents": f"{len(config.integrations)} preserved",
            "memory_mode": config.memory_mode,
        },
    }


__all__ = ["configure"]
