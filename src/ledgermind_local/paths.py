"""Path helpers for the local service."""

from __future__ import annotations

import os
import re
from pathlib import Path


class ServicePaths:
    """Explicit filesystem layout for local runtime data."""

    _DEFAULT_HOME = "~/.ledgermind/local"
    _WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")

    def __init__(self, home: str | Path | None = None) -> None:
        raw_home = os.fspath(home) if home is not None else os.environ.get(
            "LEDGERMIND_HOME",
            self._DEFAULT_HOME,
        )
        normalized_home = self._normalize_home(raw_home)
        self._home = normalized_home

    @classmethod
    def _normalize_home(cls, raw_home: str) -> Path:
        expanded = os.path.expanduser(raw_home)
        if cls._WINDOWS_ABS_RE.match(expanded):
            return Path(expanded)
        path = Path(expanded)
        if path.is_absolute():
            return path
        return path.absolute()

    @property
    def home(self) -> Path:
        return self._home

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    @property
    def token_file(self) -> Path:
        return self.home / "server.token"

    @property
    def database_file(self) -> Path:
        return self.home / "ledgermind.db"

    def resolve_database_path(self, configured_path: str | Path) -> Path:
        path = Path(configured_path).expanduser()
        return path if path.is_absolute() else self.home / path

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def service_lock_file(self) -> Path:
        return self.home / "service.lock"

    @property
    def service_pid_file(self) -> Path:
        return self.home / "service.pid"

    @property
    def backups_dir(self) -> Path:
        return self.home / "backups"
