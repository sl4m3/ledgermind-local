"""On-demand Local/Core/embedding supervisor backed by durable leases."""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Sequence

from ledgermind_local.installer.lock import InstallerLock
from ledgermind_local.installer.paths import InstallerPaths

from .endpoint import DEFAULT_ENDPOINT, validate_endpoint
from .leases import LeaseStore
from .process import ProcessManager
from .shutdown import should_shutdown
from .state import load_state, write_state


class RuntimeSupervisor:
    def __init__(
        self,
        paths: InstallerPaths,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        idle_shutdown_seconds: float = 60.0,
        lease_ttl_seconds: float = 30.0,
        commands: dict[str, Sequence[str]] | None = None,
        embedding_command: Sequence[str] | None = None,
    ) -> None:
        self.paths = paths
        self.endpoint = validate_endpoint(endpoint)
        self.idle_shutdown_seconds = max(float(idle_shutdown_seconds), 0.0)
        self.leases = LeaseStore(paths.leases_dir, ttl_seconds=lease_ttl_seconds)
        self.embedding_command = (
            tuple(str(part) for part in embedding_command)
            if embedding_command
            else None
        )
        self.commands = (
            dict(commands) if commands is not None else self._default_commands()
        )
        if self.embedding_command is not None:
            self.commands.setdefault("embedding", self.embedding_command)
        self.processes = ProcessManager()

    def _default_commands(self) -> dict[str, Sequence[str]]:
        raw = os.environ.get("LEDGERMIND_LOCAL_COMMAND", "").strip()
        if raw:
            commands: dict[str, Sequence[str]] = {"local": tuple(shlex.split(raw))}
        else:
            commands = {}
        local = self.paths.current_link / "bin" / "ledgermind-local"
        if "local" not in commands and local.is_file() and os.access(local, os.X_OK):
            commands["local"] = (str(local), "serve")
        if self.embedding_command is not None:
            commands["embedding"] = self.embedding_command
        return commands

    @staticmethod
    def _live_records(records: object) -> dict[str, object]:
        if not isinstance(records, dict):
            return {}
        live: dict[str, object] = {}
        for name, raw in records.items():
            if not isinstance(raw, dict):
                continue
            try:
                pid = int(raw["pid"])
                os.kill(pid, 0)
            except (
                KeyError,
                TypeError,
                ValueError,
                ProcessLookupError,
                PermissionError,
                OSError,
            ):
                continue
            live[str(name)] = dict(raw)
        return live

    def _write(self, *, running: bool, last_release_at: float | None = None) -> None:
        current = load_state(self.paths.runtime_state)
        own_processes = {
            name: item.as_dict() for name, item in self.processes.running().items()
        }
        processes = own_processes or self._live_records(current.get("processes", {}))
        current.update(
            {
                "schema_version": 1,
                "running": running,
                "endpoint": self.endpoint if running else None,
                "active_leases": len(self.leases.active()),
                "processes": processes,
                "last_transition": time.time(),
            }
        )
        if last_release_at is not None:
            current["last_release_at"] = last_release_at
        write_state(self.paths.runtime_state, current)

    def _start(self) -> None:
        self.processes.start(self.commands)
        self._write(running=True)

    def _stop(self, *, force: bool = False) -> dict[str, object]:
        if not force and self.leases.active():
            raise RuntimeError("runtime has active leases")
        state = load_state(self.paths.runtime_state)
        stopped = self.processes.stop()
        if not stopped:
            stopped = ProcessManager.stop_external(
                self._live_records(state.get("processes", {}))
            )
        self._write(running=False, last_release_at=time.time())
        return {"stopped": True, "processes": stopped}

    def acquire(self, *, client: str, session_id: str) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            self.leases.reap_expired()
            active_before = self.leases.active()
            state = load_state(self.paths.runtime_state)
            recorded_processes = self._live_records(state.get("processes", {}))
            started = (
                not active_before
                or not bool(state.get("running"))
                or (bool(self.commands) and not recorded_processes)
            )
            lease = self.leases.acquire(client=client, session_id=session_id)
            if started:
                self._start()
            else:
                self._write(running=True)
            return {
                "lease_id": lease.lease_id,
                "endpoint": self.endpoint,
                "expires_at": lease.as_dict()["expires_at"],
                "started": started,
            }

    def heartbeat(self, lease_id: str) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            lease = self.leases.heartbeat(lease_id)
            state = load_state(self.paths.runtime_state)
            if not state.get("running") or (
                bool(self.commands)
                and not self._live_records(state.get("processes", {}))
            ):
                self._start()
            self._write(running=True)
            return {**lease.as_dict(), "endpoint": self.endpoint}

    def release(self, lease_id: str) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            released = self.leases.release(lease_id)
            active = self.leases.active()
            stopped = False
            if not active:
                current = load_state(self.paths.runtime_state)
                last_release = float(current.get("last_release_at") or time.time())
                if should_shutdown(
                    active_leases=0,
                    leased_tasks=0,
                    pending_writes=0,
                    last_release_at=last_release,
                    idle_grace_seconds=self.idle_shutdown_seconds,
                    now=time.time(),
                ):
                    self._stop(force=True)
                    stopped = True
                else:
                    self._write(
                        running=bool(current.get("running", True)),
                        last_release_at=time.time(),
                    )
            else:
                self._write(running=True)
            return {
                "released": released,
                "active_leases": len(active),
                "stopped": stopped,
            }

    def status(self) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            expired = self.leases.reap_expired()
            active = self.leases.active()
            state = load_state(self.paths.runtime_state)
            running = bool(state.get("running"))
            if (
                running
                and not active
                and should_shutdown(
                    active_leases=0,
                    last_release_at=float(state.get("last_release_at") or time.time()),
                    idle_grace_seconds=self.idle_shutdown_seconds,
                )
            ):
                self._stop(force=True)
                state = load_state(self.paths.runtime_state)
            return {
                "running": bool(state.get("running")),
                "endpoint": state.get("endpoint"),
                "active_leases": [lease.as_dict() for lease in active],
                "expired_leases": list(expired),
                "processes": state.get("processes", {}),
            }

    def stop(self, *, force: bool = False) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            return self._stop(force=force)


__all__ = ["RuntimeSupervisor"]
