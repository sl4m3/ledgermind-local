"""On-demand Local/Core/embedding supervisor backed by durable leases."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ledgermind_local.inference.provider_telemetry import TELEMETRY_ENV
from ledgermind_local.installer.lock import InstallerLock
from ledgermind_local.installer.paths import InstallerPaths

from .endpoint import DEFAULT_ENDPOINT, validate_endpoint
from .leases import LeaseStore
from .process import ProcessManager
from .shutdown import should_shutdown
from .state import load_state, write_state


@dataclass(frozen=True, slots=True)
class RuntimeActivity:
    """Content-free work counts that gate idle shutdown."""

    known: bool
    leased_tasks: int = 0
    pending_writes: int = 0
    error_code: str | None = None

    @property
    def quiescent(self) -> bool:
        return self.known and self.leased_tasks == 0 and self.pending_writes == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "known": self.known,
            "quiescent": self.quiescent,
            "leased_tasks": self.leased_tasks,
            "pending_writes": self.pending_writes,
            "error_code": self.error_code,
        }


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
        activity_probe: Callable[[], Mapping[str, object]] | None = None,
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
        self.activity_probe = activity_probe
        self.commands = (
            dict(commands) if commands is not None else self._default_commands()
        )
        if self.embedding_command is not None:
            self.commands.setdefault("embedding", self.embedding_command)
        # Provider telemetry is part of the installed runtime's local audit
        # trail.  The child environment is deliberately sanitized, so pass
        # the content-free journal path explicitly instead of relying on an
        # inherited shell variable.
        self.processes = ProcessManager(
            environment_overrides={
                TELEMETRY_ENV: str(paths.logs_dir / "provider-telemetry.jsonl")
            }
        )

    @staticmethod
    def _live_pid(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            if isinstance(value, int):
                pid = value
            elif isinstance(value, str):
                pid = int(value)
            else:
                return None
            if pid <= 0:
                return None
            os.kill(pid, 0)
        except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
            return None
        return pid

    def _idle_reaper_command(self) -> tuple[str, ...] | None:
        """Return the installed CLI command used by the detached idle watcher."""

        installer = self.paths.current_link / "bin" / "ledgermind"
        if not installer.is_file() or not os.access(installer, os.X_OK):
            return None
        command = [str(installer), "runtime", "_idle-reap"]
        if self.paths.home_override is not None:
            command.extend(("--home", str(self.paths.home_override)))
        return tuple(command)

    @staticmethod
    def _reaper_environment() -> dict[str, str]:
        environment: dict[str, str] = {}
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "HOME",
            "USER",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
        ):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment

    def _ensure_idle_reaper(self) -> int | None:
        """Keep exactly one detached watcher for an installed runtime."""

        state = load_state(self.paths.runtime_state)
        existing = self._live_pid(state.get("idle_reaper_pid"))
        if existing is not None:
            return existing
        command = self._idle_reaper_command()
        if command is None:
            return None
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            env=self._reaper_environment(),
        )
        state = load_state(self.paths.runtime_state)
        state["idle_reaper_pid"] = process.pid
        state["last_transition"] = time.time()
        write_state(self.paths.runtime_state, state)
        return process.pid

    def _set_idle_since(self, value: float | None) -> None:
        state = load_state(self.paths.runtime_state)
        if value is None:
            state.pop("idle_since", None)
        else:
            state["idle_since"] = value
        state["last_transition"] = time.time()
        write_state(self.paths.runtime_state, state)

    @staticmethod
    def _parse_activity(payload: Mapping[str, object]) -> RuntimeActivity:
        known = payload.get("known") is True

        def count(name: str) -> int:
            value = payload.get(name, 0)
            if isinstance(value, bool):
                raise TypeError(f"runtime activity {name} must be an integer")
            if isinstance(value, int):
                parsed = value
            elif isinstance(value, str):
                parsed = int(value)
            else:
                raise TypeError(f"runtime activity {name} must be an integer")
            if parsed < 0:
                raise ValueError(f"runtime activity {name} must not be negative")
            return parsed

        return RuntimeActivity(
            known=known,
            leased_tasks=count("leased_tasks"),
            pending_writes=count("pending_writes"),
            error_code=(
                str(payload["error_code"])
                if payload.get("error_code") is not None
                else None
            ),
        )

    def _activity_snapshot(self) -> RuntimeActivity:
        """Read fresh activity; an unavailable probe blocks automatic shutdown."""

        try:
            if self.activity_probe is not None:
                return self._parse_activity(self.activity_probe())
            local_command = self.commands.get("local")
            if (
                not local_command
                or os.path.basename(str(local_command[0])) != "ledgermind-local"
            ):
                return RuntimeActivity(known=True)
            token_file = self.paths.data_dir / "local" / "server.token"
            token = token_file.read_text(encoding="utf-8").strip()
            if not token:
                raise RuntimeError("Local API token is empty")
            request = Request(
                self.endpoint + "/runtime/activity",
                headers={
                    "authorization": f"Bearer {token}",
                    "accept": "application/json",
                },
                method="GET",
            )
            with urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("runtime activity response must be an object")
            return self._parse_activity(payload)
        except Exception:  # noqa: BLE001 - unknown activity must fail closed
            return RuntimeActivity(known=False, error_code="activity_unavailable")

    def _record_activity(self, activity: RuntimeActivity) -> None:
        state = load_state(self.paths.runtime_state)
        state["activity"] = activity.as_dict()
        state["last_transition"] = time.time()
        write_state(self.paths.runtime_state, state)

    def _update_idle_boundary(
        self,
        *,
        active_leases: int,
        now: float,
    ) -> tuple[RuntimeActivity, float | None]:
        state = load_state(self.paths.runtime_state)
        if active_leases:
            if state.get("idle_since") is not None:
                self._set_idle_since(None)
            return RuntimeActivity(known=True), None
        activity = self._activity_snapshot()
        self._record_activity(activity)
        state = load_state(self.paths.runtime_state)
        if not activity.quiescent:
            if state.get("idle_since") is not None:
                self._set_idle_since(None)
            return activity, None
        idle_since = state.get("idle_since")
        if not isinstance(idle_since, int | float):
            idle_since = now
            self._set_idle_since(idle_since)
        return activity, float(idle_since)

    def _default_commands(self) -> dict[str, Sequence[str]]:
        raw = os.environ.get("LEDGERMIND_LOCAL_COMMAND", "").strip()
        if raw:
            commands: dict[str, Sequence[str]] = {"local": tuple(shlex.split(raw))}
        else:
            commands = {}
        local = self.paths.current_link / "bin" / "ledgermind-local"
        if "local" not in commands and local.is_file() and os.access(local, os.X_OK):
            commands["local"] = (
                str(local),
                "--home",
                str(self.paths.data_dir / "local"),
                "serve",
            )
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
        processes = {
            **self._live_records(current.get("processes", {})),
            **own_processes,
        }
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

    def _all_commands_running(self, records: object) -> bool:
        if not self.commands:
            return True
        live = self._live_records(records)
        return set(self.commands).issubset(live)

    def _start(self) -> None:
        self.processes.start(self.commands)
        try:
            self._wait_for_owned_local()
        except BaseException:
            self.processes.stop()
            self._write(running=False)
            raise
        self._write(running=True)

    def _wait_for_owned_local(self, *, timeout_seconds: float = 30.0) -> None:
        """Verify that the authenticated endpoint belongs to this installation."""

        if "local" not in self.commands:
            return
        local_command = self.commands["local"]
        if (
            not local_command
            or os.path.basename(str(local_command[0])) != "ledgermind-local"
        ):
            time.sleep(0.05)
            if self.processes.running():
                return
            raise RuntimeError("managed Local process exited during startup")
        token_file = self.paths.data_dir / "local" / "server.token"
        if not token_file.is_file():
            raise RuntimeError("Local API token is missing; run LedgerMind configure")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError("Local API token is empty; run LedgerMind configure")
        deadline = time.monotonic() + max(float(timeout_seconds), 0.1)
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if not self.processes.running():
                break
            request = Request(
                self.endpoint + "/ping",
                headers={
                    "authorization": f"Bearer {token}",
                    "accept": "application/json",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=0.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict) and payload.get("pong") == "true":
                    return
                last_error = RuntimeError("unexpected authenticated ping response")
            except HTTPError as exc:
                last_error = exc
                if exc.code in {401, 403}:
                    raise RuntimeError(
                        f"runtime endpoint conflict at {self.endpoint}: "
                        "another service is using the configured address"
                    ) from exc
            except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
            time.sleep(0.05)
        raise RuntimeError(
            f"LedgerMind Local failed to become ready at {self.endpoint}"
        ) from last_error

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
            state = load_state(self.paths.runtime_state)
            recorded_processes = self._live_records(state.get("processes", {}))
            started = not bool(state.get("running")) or not self._all_commands_running(
                recorded_processes
            )
            lease = self.leases.acquire(client=client, session_id=session_id)
            try:
                if started:
                    self._start()
                else:
                    self._write(running=True)
                self._set_idle_since(None)
                self._ensure_idle_reaper()
            except BaseException:
                self.leases.release(lease.lease_id)
                raise
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
                not self._all_commands_running(state.get("processes", {}))
            ):
                self._start()
            self._write(running=True)
            self._set_idle_since(None)
            self._ensure_idle_reaper()
            return {**lease.as_dict(), "endpoint": self.endpoint}

    def release(self, lease_id: str) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            released = self.leases.release(lease_id)
            active = self.leases.active()
            stopped = False
            activity: RuntimeActivity | None = None
            if not active:
                current = load_state(self.paths.runtime_state)
                released_at = time.time()
                self._write(
                    running=bool(current.get("running", True)),
                    last_release_at=released_at,
                )
                activity, idle_since = self._update_idle_boundary(
                    active_leases=0,
                    now=released_at,
                )
                if idle_since is not None and should_shutdown(
                    active_leases=0,
                    leased_tasks=activity.leased_tasks,
                    pending_writes=activity.pending_writes,
                    last_release_at=idle_since,
                    idle_grace_seconds=self.idle_shutdown_seconds,
                    now=released_at,
                ):
                    active = self.leases.active()
                    activity, idle_since = self._update_idle_boundary(
                        active_leases=len(active),
                        now=time.time(),
                    )
                    if idle_since is not None and activity.quiescent and not active:
                        self._stop(force=True)
                        stopped = True
                else:
                    self._ensure_idle_reaper()
            else:
                self._write(running=True)
                self._set_idle_since(None)
            return {
                "released": released,
                "active_leases": len(active),
                "stopped": stopped,
                "activity": activity.as_dict() if activity is not None else None,
            }

    def status(self) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            expired = self.leases.reap_expired()
            active = self.leases.active()
            state = load_state(self.paths.runtime_state)
            running = bool(state.get("running"))
            if running and not self._all_commands_running(state.get("processes", {})):
                self._write(running=False)
                state = load_state(self.paths.runtime_state)
                running = False
            activity: RuntimeActivity | None = None
            if running:
                now = time.time()
                activity, idle_since = self._update_idle_boundary(
                    active_leases=len(active),
                    now=now,
                )
                if idle_since is not None and should_shutdown(
                    active_leases=len(active),
                    leased_tasks=activity.leased_tasks,
                    pending_writes=activity.pending_writes,
                    last_release_at=idle_since,
                    idle_grace_seconds=self.idle_shutdown_seconds,
                    now=now,
                ):
                    # Re-read both lease and work state immediately before stop.
                    active = self.leases.active()
                    activity, idle_since = self._update_idle_boundary(
                        active_leases=len(active),
                        now=time.time(),
                    )
                    if idle_since is not None and activity.quiescent and not active:
                        self._stop(force=True)
                        state = load_state(self.paths.runtime_state)
            return {
                "running": bool(state.get("running")),
                "endpoint": state.get("endpoint"),
                "active_leases": [lease.as_dict() for lease in active],
                "expired_leases": list(expired),
                "processes": state.get("processes", {}),
                "activity": (
                    activity.as_dict()
                    if activity is not None
                    else state.get("activity")
                ),
            }

    def watch_idle(self, *, poll_interval_seconds: float = 0.5) -> dict[str, object]:
        """Watch leases until the installed runtime reaches its idle deadline."""

        poll_interval = max(float(poll_interval_seconds), 0.05)
        while True:
            with InstallerLock(self.paths.state_dir / "runtime.lock"):
                self.leases.reap_expired()
                active = self.leases.active()
                state = load_state(self.paths.runtime_state)
                if not state.get("running"):
                    return {"stopped": False, "reason": "runtime_not_running"}
                if not self._all_commands_running(state.get("processes", {})):
                    self._write(running=False)
                    return {"stopped": False, "reason": "runtime_process_missing"}
                now = time.time()
                activity, idle_since = self._update_idle_boundary(
                    active_leases=len(active),
                    now=now,
                )
                if idle_since is not None and should_shutdown(
                    active_leases=len(active),
                    leased_tasks=activity.leased_tasks,
                    pending_writes=activity.pending_writes,
                    last_release_at=float(idle_since),
                    idle_grace_seconds=self.idle_shutdown_seconds,
                    now=now,
                ):
                    # The lock excludes a concurrent acquire; the fresh
                    # activity read closes the final task-completion race.
                    active = self.leases.active()
                    activity, idle_since = self._update_idle_boundary(
                        active_leases=len(active),
                        now=time.time(),
                    )
                    if idle_since is not None and activity.quiescent and not active:
                        stopped = self._stop(force=True)
                        return {**stopped, "reason": "idle_timeout"}
            time.sleep(poll_interval)

    def stop(self, *, force: bool = False) -> dict[str, object]:
        with InstallerLock(self.paths.state_dir / "runtime.lock"):
            return self._stop(force=force)


__all__ = ["RuntimeSupervisor"]
