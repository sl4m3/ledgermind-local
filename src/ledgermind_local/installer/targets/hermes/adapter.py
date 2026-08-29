"""First-party Hermes TargetAdapter implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from ..base import AdapterContext, BaseTargetAdapter, TargetDiscovery
from .discover import discover_hermes
from .hooks import hook_registration
from .install import install_plugin
from .lifecycle import lifecycle_contract
from .verify import verify_hermes_plugin


class HermesTargetAdapter(BaseTargetAdapter):
    id = "hermes"
    label = "Hermes"

    def discover(self) -> TargetDiscovery:
        return discover_hermes()

    def _discovery(self, context: AdapterContext) -> TargetDiscovery:
        discovery = context.discovery or self.discover()
        if not discovery.detected or discovery.home is None:
            raise RuntimeError(discovery.detail or "Hermes was not discovered")
        return discovery

    def _home(self, context: AdapterContext) -> Path:
        home = self._discovery(context).home
        assert home is not None
        return home

    def preflight(self, context: AdapterContext) -> dict[str, Any]:
        discovery = self._discovery(context)
        return {
            "target": self.id,
            "discovery": discovery.as_dict(),
            "hooks": hook_registration(),
        }

    def install(self, context: AdapterContext) -> dict[str, Any]:
        home = self._home(context)
        if context.dry_run:
            return {
                "status": "dry_run",
                "plugin_root": str(home / "plugins" / "ledgermind-hermes"),
            }
        payload = {
            "enabled": bool(context.metadata.get("enabled", True)),
            "endpoint": "http://127.0.0.1:8765",
            "token_file": str(context.paths.data_dir / "local" / "server.token"),
            "memory_space_id": "hermes-default",
            "source_instance_id": f"hermes-{uuid4().hex}",
            "profile_id": "generation-operational",
            "state_db_path": str(home / "state.db"),
            "spool_dir": str(context.paths.integrations_dir / "hermes"),
            "runtime_endpoint": "http://127.0.0.1:8765",
            "runtime_command": str(context.paths.bin_link),
            "lease_ttl_seconds": context.config.runtime.lease_ttl_seconds,
            "heartbeat_seconds": context.config.runtime.heartbeat_seconds,
        }
        result = install_plugin(
            hermes_home=home,
            bundle_root=context.bundle_root,
            plugin_config=payload,
        )
        result["hooks"] = self.register_hooks(context)
        return result

    def configure(self, context: AdapterContext) -> dict[str, Any]:
        return self.install(context)

    def register_hooks(self, context: AdapterContext) -> dict[str, Any]:
        return {
            **hook_registration(),
            "contract": lifecycle_contract(
                endpoint="http://127.0.0.1:8765",
                lease_ttl_seconds=context.config.runtime.lease_ttl_seconds,
                heartbeat_seconds=context.config.runtime.heartbeat_seconds,
            ),
        }

    def verify(self, context: AdapterContext) -> dict[str, Any]:
        return verify_hermes_plugin(
            self._home(context) / "plugins" / "ledgermind-hermes"
        )

    def repair(self, context: AdapterContext) -> dict[str, Any]:
        return self.install(context)

    def uninstall(
        self, context: AdapterContext, *, purge: bool = False
    ) -> dict[str, Any]:
        from .uninstall import uninstall_plugin

        return uninstall_plugin(self._home(context), purge=purge)

    def runtime_environment(self, context: AdapterContext) -> dict[str, str]:
        return {
            "LEDGERMIND_HERMES_CONFIG": str(
                self._home(context) / "plugins" / "ledgermind-hermes" / "config.json"
            ),
            "LEDGERMIND_RUNTIME_ENDPOINT": "http://127.0.0.1:8765",
        }


__all__ = ["HermesTargetAdapter"]
