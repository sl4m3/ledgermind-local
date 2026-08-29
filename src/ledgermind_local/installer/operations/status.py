"""Installer status operation."""

from __future__ import annotations

from typing import Any

from ledgermind_local.runtime.supervisor import RuntimeSupervisor

from ..paths import InstallerPaths
from .integrations import integration_status


def status(*, paths: InstallerPaths) -> dict[str, Any]:
    current = str(paths.current_link.resolve()) if paths.current_link.exists() else None
    config_exists = paths.config_file.is_file()
    runtime = RuntimeSupervisor(paths).status()
    integrations: dict[str, Any] = {"status": "not_configured", "integrations": {}}
    if config_exists:
        integrations = integration_status(paths=paths)
        active_leases = runtime.get("active_leases", [])
        if not isinstance(active_leases, list):
            active_leases = []
        active_clients = {
            str(item.get("client"))
            for item in active_leases
            if isinstance(item, dict)
        }
        for target_id, item in integrations["integrations"].items():
            item["active"] = target_id in active_clients
    return {
        "installed": bool(current and config_exists),
        "current": current,
        "config": str(paths.config_file),
        "config_exists": config_exists,
        "runtime": runtime,
        "integrations": integrations["integrations"],
    }


__all__ = ["status"]
