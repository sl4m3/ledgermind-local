"""Managed child processes for on-demand Local/Core services."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class ManagedProcess:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen[bytes]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pid": self.process.pid,
            "command": list(self.command),
        }


class ProcessManager:
    def __init__(self) -> None:
        self.processes: dict[str, ManagedProcess] = {}

    def start(self, commands: dict[str, Sequence[str]]) -> dict[str, ManagedProcess]:
        for name, command in commands.items():
            if not command or name in self.processes:
                continue
            process = subprocess.Popen(
                [str(part) for part in command],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                env=_clean_environment(),
            )
            self.processes[name] = ManagedProcess(
                name, tuple(str(part) for part in command), process
            )
        return dict(self.processes)

    def running(self) -> dict[str, ManagedProcess]:
        dead = [
            name
            for name, item in self.processes.items()
            if item.process.poll() is not None
        ]
        for name in dead:
            self.processes.pop(name, None)
        return dict(self.processes)

    def stop(self, *, timeout_seconds: float = 5.0) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        for name, item in list(self.processes.items())[::-1]:
            if item.process.poll() is None:
                item.process.terminate()
                try:
                    item.process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    item.process.kill()
                    item.process.wait(timeout=timeout_seconds)
            result[name] = item.process.returncode
            self.processes.pop(name, None)
        return result

    @staticmethod
    def stop_external(
        records: dict[str, object], *, timeout_seconds: float = 5.0
    ) -> dict[str, int | None]:
        """Stop processes recorded by a previous supervisor instance."""

        result: dict[str, int | None] = {}
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
                result[name] = None
                continue
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                result[name] = None
                continue
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError:
                    break
                time.sleep(0.05)
            try:
                os.kill(pid, 0)
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                result[name] = None
            result[name] = None
        return result


def _clean_environment() -> dict[str, str]:
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
        "PYTHONPATH",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


__all__ = ["ManagedProcess", "ProcessManager"]
