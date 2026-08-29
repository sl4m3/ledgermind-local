"""Safe uninstall preserving user memory/config by default."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ...runtime.supervisor import RuntimeSupervisor
from ..config_writer import select_secret_backend
from ..paths import InstallerPaths
from ..targets.base import AdapterContext
from ..targets.registry import get_target_adapter
from .common import remove_bin_link


def uninstall(
    *,
    paths: InstallerPaths,
    purge_data: bool = False,
    purge_config: bool = False,
    yes: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if (purge_data or purge_config) and not yes:
        raise ValueError("--yes is required with purge flags")
    if dry_run:
        return {
            "status": "dry_run",
            "purge_data": purge_data,
            "purge_config": purge_config,
            "preserved_data": not purge_data,
            "preserved_config": not purge_config,
        }
    config = None
    if paths.config_file.is_file():
        from ..config_writer import load_installer_config

        config = load_installer_config(paths.config_file)
    RuntimeSupervisor(paths).stop(force=True)
    adapter_result: dict[str, Any] = {}
    if config is not None:
        for selected in config.integrations:
            adapter = get_target_adapter(selected.id)
            discovery = adapter.discover()
            if discovery.detected:
                adapter_result[selected.id] = adapter.uninstall(
                    AdapterContext(config=config, paths=paths, discovery=discovery),
                    purge=False,
                )
    remove_bin_link(paths)
    if paths.current_link.is_symlink():
        paths.current_link.unlink(missing_ok=True)
    releases_removed = 0
    if paths.releases_dir.exists():
        for release in paths.releases_dir.iterdir():
            if release.is_dir():
                shutil.rmtree(release)
                releases_removed += 1
    if purge_data:
        shutil.rmtree(paths.memory_data_dir, ignore_errors=True)
        shutil.rmtree(paths.models_dir, ignore_errors=True)
        shutil.rmtree(paths.integrations_dir, ignore_errors=True)
        if config is not None and config.memory_data_path:
            custom_memory = Path(config.memory_data_path).expanduser()
            if custom_memory != paths.memory_data_dir:
                shutil.rmtree(custom_memory, ignore_errors=True)
    if purge_config:
        backend = select_secret_backend(paths)
        backend.delete("generation/token")
        backend.delete("embedding/token")
        shutil.rmtree(paths.config_dir, ignore_errors=True)
        shutil.rmtree(paths.data_dir / "local", ignore_errors=True)
    return {
        "status": "passed",
        "integrations": adapter_result,
        "releases_removed": releases_removed,
        "preserved_data": not purge_data,
        "preserved_config": not purge_config,
    }
