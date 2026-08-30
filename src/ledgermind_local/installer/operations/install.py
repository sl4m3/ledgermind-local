"""Transactional install orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config_writer import (
    build_local_config,
    resolve_provider_tokens,
    write_installer_config,
    write_local_config,
    write_local_profiles,
)
from ..embeddings.model_download import download_model
from ..embeddings.runtime_install import install_runtime
from ..embeddings.verification import verify_model_files
from ..errors import ConfigurationError, InstallerError
from ..lock import InstallerLock
from ..manifest import InstallManifest
from ..models import InstallerConfig
from ..paths import InstallerPaths
from ..preflight import check_preflight
from ..profiles.embedding_local import build_local_embedding_plan
from ..profiles.generation import build_generation_profiles
from ..profiles.probes import probe_embedding_api, probe_generation
from ..transaction import InstallTransaction
from ..verify import (
    public_key_from_environment,
    sha256_file,
    verify_artifact_signature,
    verify_bundle_tree,
    verify_file,
)
from .common import (
    fetch_bundle,
    fetch_release,
    install_bin_link,
    platform_manifest,
    resolve_manifest,
    unpack_bundle,
    verify_manifest_signature,
    write_install_history,
)
from .doctor import doctor
from .integrations import connect_integration


def install_plan(
    config: InstallerConfig,
    *,
    paths: InstallerPaths,
    manifest: InstallManifest | None = None,
) -> dict[str, Any]:
    profiles = list(build_generation_profiles(config.generation))
    if config.embedding.mode == "api" and config.embedding.api is not None:
        profiles.append(
            {
                "profile_id": "embedding-default",
                "slot": "embedding",
                "endpoint": config.embedding.api.endpoint,
                "model": config.embedding.api.model,
                "dimensions": config.embedding.api.dimensions,
            }
        )
    else:
        assert config.embedding.local is not None
        if manifest is None:
            raise ConfigurationError(
                "local embeddings require a signed install manifest"
            )
        entry = manifest.catalog_entry(config.embedding.local.catalog_id)
        profiles.append(build_local_embedding_plan(config.embedding.local, entry))
    return {
        "integrations": [item.model_dump(mode="json") for item in config.integrations],
        "memory_mode": config.memory_mode,
        "platform": _platform_name(),
        "memory_data_path": config.memory_data_path or str(paths.memory_data_dir),
        "profiles": profiles,
        "runtime": config.runtime.model_dump(mode="json"),
        "release_version": manifest.release_version if manifest else None,
        "requires_signed_bundle": True,
    }


def _platform_name() -> str:
    from ..preflight import platform_id

    return platform_id()


def install(
    *,
    config: InstallerConfig,
    paths: InstallerPaths,
    manifest_path: str | Path | None = None,
    bundle: str | Path | None = None,
    manifest_signature: str | Path | None = None,
    public_key: bytes | str | None = None,
    dry_run: bool = False,
    skip_provider_probe: bool = False,
    bundle_root_override: str | Path | None = None,
    generation_stdin: str | None = None,
    embedding_stdin: str | None = None,
) -> dict[str, Any]:
    """Install one release and return safe operation metadata."""

    paths.ensure()
    manifest, bundle_root = resolve_manifest(
        manifest_path, bundle_root_override or bundle
    )
    if dry_run:
        plan = install_plan(config, paths=paths, manifest=manifest)
        plan["status"] = "dry_run"
        return plan
    if manifest is None:
        fetched_manifest, fetched_signature, fetched_bundle = fetch_release(
            paths,
            public_key=public_key or public_key_from_environment(),
        )
        manifest_path = fetched_manifest
        manifest_signature = fetched_signature
        bundle_root = fetched_bundle
        manifest, _ = resolve_manifest(manifest_path, bundle_root)
    if manifest is None:
        raise ConfigurationError("a signed install manifest is required")
    manifest_file = (
        Path(manifest_path).expanduser()
        if manifest_path
        else (Path(bundle_root) / "install-manifest.json" if bundle_root else None)
    )
    signature_file = (
        Path(manifest_signature).expanduser()
        if manifest_signature
        else (manifest_file.with_suffix(".sig") if manifest_file else None)
    )
    if manifest_file is None or signature_file is None or not signature_file.is_file():
        raise ConfigurationError("manifest signature is required")
    verify_manifest_signature(
        manifest_file, signature_file, public_key or public_key_from_environment()
    )
    if bundle_root is None:
        bundle_root = fetch_bundle(
            paths,
            manifest,
            public_key=public_key or public_key_from_environment(),
        )
    platform_manifest(manifest)
    preflight = check_preflight(paths)
    plan = install_plan(config, paths=paths, manifest=manifest)
    bundle_source = Path(bundle_root)
    if bundle_source.is_file():
        bundle_record = platform_manifest(manifest).bundle
        verify_file(bundle_source, bundle_record)
        verify_artifact_signature(
            bundle_source,
            bundle_record,
            public_key=public_key or public_key_from_environment(),
        )
    bundle_path = unpack_bundle(bundle_root, paths.staging_dir / "bundle-unpacked")
    plan["bundle_verification"] = verify_bundle_tree(
        bundle_path, public_key=public_key or public_key_from_environment()
    )
    local_embedding_metadata: dict[str, Any] | None = None
    effective_config = config
    if config.embedding.mode == "local" and config.embedding.local is not None:
        entry = manifest.catalog_entry(config.embedding.local.catalog_id)
        local_plan = build_local_embedding_plan(config.embedding.local, entry)
        effective_config = config.model_copy(
            update={
                "embedding": config.embedding.model_copy(
                    update={
                        "local": config.embedding.local.model_copy(
                            update={"device": local_plan["device"]}
                        )
                    }
                )
            }
        )
        if config.embedding.local.model_path is not None:
            model_path = Path(config.embedding.local.model_path).expanduser()
            if not model_path.is_file():
                raise ConfigurationError(
                    f"custom embedding model is missing: {model_path}"
                )
            catalog_compatibility = entry.get("runtime_compatibility")
            configured_compatibility = config.advanced.custom_runtime_compatibility
            if (
                catalog_compatibility is not None
                and str(catalog_compatibility) != configured_compatibility
            ):
                raise ConfigurationError(
                    "custom embedding runtime compatibility does not match catalog"
                )
            expected_digest = config.advanced.custom_model_sha256
            if (
                expected_digest is None
                or sha256_file(model_path).lower() != expected_digest.lower()
            ):
                raise ConfigurationError(
                    "custom embedding model SHA-256 does not match configuration"
                )
            local_plan["model"] = {
                "status": "passed",
                "source": "advanced-custom-model",
                "files_checked": 1,
                "sha256": expected_digest,
                "license": entry.get("license"),
            }
        else:
            model_dir = download_model(entry, paths)
            local_plan["model"] = verify_model_files(model_dir, entry)
        local_plan["runtime"] = install_runtime(
            entry,
            device=str(local_plan["device"]),
            bundle_root=bundle_path,
            paths=paths,
        )
        local_embedding_metadata = local_plan
    if not skip_provider_probe:
        generation_token, embedding_token = resolve_provider_tokens(
            config,
            paths,
            generation_stdin=generation_stdin,
            embedding_stdin=embedding_stdin,
        )
        plan["generation_probe"] = probe_generation(
            config.generation, token=generation_token
        )
        if config.embedding.mode == "api" and config.embedding.api is not None:
            assert embedding_token is not None
            plan["embedding_probe"] = probe_embedding_api(
                config.embedding.api, token=embedding_token
            )
    release_dir = paths.release_dir(manifest.release_version)
    platform_config = effective_config.model_copy(update={"integrations": ()})
    local_database_path = Path(
        build_local_config(platform_config, paths).rounds_database_path
    )
    doctor_report: dict[str, Any] = {}
    with (
        InstallerLock(paths.install_lock),
        InstallTransaction(
            paths, release_version=manifest.release_version
        ) as transaction,
    ):
        staged = transaction.copy_bundle(bundle_path)
        transaction.write_marker(
            {
                "release_version": manifest.release_version,
                "platform": preflight["platform"],
            }
        )
        transaction.commit(staged_bundle=staged, switch_current=False)
        try:
            for path in (
                paths.config_file,
                paths.profiles_file,
                paths.secrets_file,
                paths.config_dir / "local-config.json",
                paths.data_dir / "local" / "config.json",
                local_database_path,
                Path(f"{local_database_path}-wal"),
                Path(f"{local_database_path}-shm"),
            ):
                transaction.backup_file(path)
            config_metadata = write_installer_config(
                platform_config,
                paths,
                generation_stdin=generation_stdin,
                embedding_stdin=embedding_stdin,
            )
            config_metadata["local_config"] = str(
                write_local_config(platform_config, paths, release_dir=release_dir)
            )
            config_metadata["local_profiles"] = write_local_profiles(
                platform_config, paths
            )
            transaction.switch_current()
            install_bin_link(paths)
            doctor_report = doctor(
                paths=paths,
                full_smoke=True,
                probe_providers=not skip_provider_probe,
                verify_integrations=False,
            )
            if doctor_report.get("status") != "passed":
                raise InstallerError("post-install doctor or sandbox smoke failed")
        except Exception:
            transaction.rollback()
            transaction.restore_backups()
            raise
    integration_results: dict[str, Any] = {}
    integration_failures: list[str] = []
    for selected in effective_config.integrations:
        try:
            integration_results[selected.id] = connect_integration(
                paths=paths, target_id=selected.id, enabled=selected.enabled
            )
        except Exception as exc:  # noqa: BLE001
            integration_results[selected.id] = {
                "status": "failed",
                "error": str(exc),
            }
            integration_failures.append(selected.id)
    write_install_history(
        paths,
        {
            "operation": "install",
            "release_version": manifest.release_version,
            "platform": preflight["platform"],
            "timestamp": time.time(),
            "current": str(paths.current_link),
        },
    )
    return {
        "status": "partial" if integration_failures else "success",
        "release_version": manifest.release_version,
        "platform": preflight["platform"],
        "preflight": preflight,
        "plan": plan,
        "local_embedding": local_embedding_metadata,
        "config": config_metadata,
        "integrations": integration_results,
        "integration_failures": integration_failures,
        "doctor": doctor_report,
        "smoke_test": doctor_report.get("smoke_test", {}),
        "current": str(paths.current_link),
        "release_dir": str(release_dir),
        "doctor_required": True,
        "smoke_test_required": True,
    }


__all__ = ["install", "install_plan"]
