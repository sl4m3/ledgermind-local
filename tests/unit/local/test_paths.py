"""Tests for ServicePaths and bootstrap directory layout behavior."""

from __future__ import annotations

from pathlib import Path

import os
import pytest

from paths import ServicePaths
from bootstrap import bootstrap_local_service
from config import LocalConfig


def test_default_home_uses_explicit_constant() -> None:
    os.environ.pop("LEDGERMIND_HOME", None)
    paths = ServicePaths()
    assert paths.home == Path("~/.ledgermind/local").expanduser()


def test_env_home_is_respected_when_not_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    home = "/tmp/ledgermind-env-home"
    monkeypatch.setenv("LEDGERMIND_HOME", home)
    paths = ServicePaths()
    assert paths.home == Path(home)


def test_explicit_home_has_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LEDGERMIND_HOME", "/tmp/ledgermind-env-home")
    explicit = tmp_path / "explicit"
    paths = ServicePaths(home=explicit)
    assert paths.home == explicit.absolute()
    assert paths.home.is_absolute()


def test_windows_like_path_is_handled_without_posix_assumptions() -> None:
    windows_like = r"C:\Users\test\ledgermind"
    paths = ServicePaths(home=windows_like)
    assert str(paths.home).endswith(r"Users\test\ledgermind")


def test_bootstrap_creates_private_directories(tmp_path: Path) -> None:
    service_dir = tmp_path / "secure"
    paths, _ = bootstrap_local_service(home=service_dir, config=LocalConfig(config_version=1))

    assert paths.home.exists()
    assert paths.logs_dir.exists()

    home_mode = paths.home.stat().st_mode & 0o777
    logs_mode = paths.logs_dir.stat().st_mode & 0o777
    assert home_mode <= 0o700
    assert logs_mode <= 0o700
