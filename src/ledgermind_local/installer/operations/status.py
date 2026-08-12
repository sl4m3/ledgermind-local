"""Installer status operation."""

from __future__ import annotations

from typing import Any

from ledgermind_local.runtime.supervisor import RuntimeSupervisor

from ..paths import InstallerPaths


def status(*, paths: InstallerPaths) -> dict[str, Any]:
    current = str(paths.current_link.resolve()) if paths.current_link.exists() else None
    config_exists = paths.config_file.is_file()
    runtime = RuntimeSupervisor(paths).status()
    return {
        "installed": bool(current and config_exists),
        "current": current,
        "config": str(paths.config_file),
        "config_exists": config_exists,
        "runtime": runtime,
        "hermes_integrations": str(paths.integrations_dir),
    }


__all__ = ["status"]
