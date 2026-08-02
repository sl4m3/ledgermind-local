"""Tests for local `ledgermind serve` command."""

from __future__ import annotations

import os
import signal
from pathlib import Path

from argparse import Namespace

from config import LocalConfig
from bootstrap import initialize_local_layout
from cli import (
    _command_serve,
    _coalesce_optional,
    _install_signal_handlers,
    _restore_signal_handlers,
)
from service_lock import ServiceLockError
import cli as cli_module


def test_coalesce_optional_returns_fallback() -> None:
    assert _coalesce_optional(None, "default") == "default"
    assert _coalesce_optional("", "default") == "default"
    assert _coalesce_optional(0, "default") == 0
    assert _coalesce_optional("0.0.0.0", "default") == "0.0.0.0"


def test_install_signal_handlers_restores_after_restore() -> None:
    server = type("_S", (), {"should_exit": False})()
    original_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    installed = _install_signal_handlers(server)
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert server.should_exit is True
    finally:
        _restore_signal_handlers(installed)

    assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]
    assert signal.getsignal(signal.SIGTERM) == original_handlers[signal.SIGTERM]


def test_command_serve_rejects_remote_host_without_allow_remote_bind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "service"
    initialize_local_layout(
        home=home,
        config=LocalConfig(config_version=1, bind_host="127.0.0.1", allow_remote_bind=False),
    )

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("lock should never be created for rejected bind host")

        def __enter__(self) -> "DummyLock":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(cli_module, "ServiceLock", DummyLock)

    args = Namespace(home=str(home), host="0.0.0.0", port=None, reload=False)
    code = _command_serve(args)
    assert code == 2


def test_command_serve_reports_lock_error(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "service"
    initialize_local_layout(home=home)

    class FailingLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "FailingLock":
            raise ServiceLockError("service is already running")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(cli_module, "ServiceLock", FailingLock)
    events: list[str] = []

    def fake_open_db_connection(_path: Path) -> None:
        events.append("open_db")
        raise RuntimeError("should not open db")

    monkeypatch.setattr(cli_module, "open_sqlite_connection", fake_open_db_connection)

    args = Namespace(home=str(home), host=None, port=None, reload=False)
    code = _command_serve(args)

    assert code == 1
    assert events == []


def test_command_serve_allows_remote_host_when_explicitly_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "service"
    initialize_local_layout(
        home=home,
        config=LocalConfig(config_version=1, allow_remote_bind=True, bind_host="127.0.0.1"),
    )

    events: list[str] = []

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append(f"lock_enter:{args[0]}")

        def __enter__(self) -> "DummyLock":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append("lock_exit")

    class DummyServer:
        def __init__(self) -> None:
            self.should_exit = False
            self.run_calls = 0

        def run(self) -> None:
            events.append("server_run")

    def fake_server_builder(*, app, host: str, port: int, reload: bool) -> DummyServer:
        events.append(f"server_builder:{host}:{port}:{reload}")
        return DummyServer()

    monkeypatch.setattr(cli_module, "ServiceLock", DummyLock)
    monkeypatch.setattr(cli_module, "_build_uvicorn_server", fake_server_builder)
    monkeypatch.setattr(cli_module, "_write_pid_file", lambda path, pid: events.append("pid_write"))
    monkeypatch.setattr(cli_module, "_remove_pid_file", lambda path: events.append("pid_remove"))
    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host="0.0.0.0", port=None, reload=False)
    code = _command_serve(args)

    assert code == 0
    assert "server_builder:0.0.0.0:8765:False" in events
    assert events[0].startswith("lock_enter:")
    assert "server_run" in events
def test_command_serve_writes_pid_and_starts_server(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "service"
    initialize_local_layout(home=home)
    token = (home / "server.token").read_text(encoding="utf-8").strip()
    assert token

    events: list[str] = []

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.path = args[0] if args else None

        def __enter__(self) -> "DummyLock":
            events.append(f"lock_enter:{self.path}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(f"lock_exit:{self.path}")

    class DummyServer:
        def __init__(self) -> None:
            self.should_exit = False
            self.run_calls = 0

        def run(self) -> None:
            self.run_calls += 1
            events.append("server_run")

    created = {"ok": False}

    def fake_server_builder(*, app, host: str, port: int, reload: bool) -> DummyServer:
        created["ok"] = True
        events.append(f"server_builder:{host}:{port}:{reload}")
        return DummyServer()

    def fake_write_pid(path: Path, pid: int) -> None:
        events.append(f"pid_write:{path}:{pid}")

    def fake_remove_pid(path: Path) -> None:
        events.append(f"pid_remove:{path}")

    monkeypatch.setattr(cli_module, "ServiceLock", DummyLock)
    monkeypatch.setattr(cli_module, "_build_uvicorn_server", fake_server_builder)
    monkeypatch.setattr(cli_module, "_write_pid_file", fake_write_pid)
    monkeypatch.setattr(cli_module, "_remove_pid_file", fake_remove_pid)
    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host=None, port=None, reload=False)
    code = _command_serve(args)

    assert code == 0
    assert created["ok"] is True
    assert events[0].startswith("lock_enter:")
    assert events[1].startswith("pid_write:") and events[1].endswith("12345")
    assert "server_builder:127.0.0.1:8765:False" in events
    assert "server_run" in events
    assert any(item.startswith("lock_exit:") for item in events)
    assert any(item.startswith("pid_remove:") for item in events)


def test_command_serve_applies_migrations_before_starting_server(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "service"
    initialize_local_layout(home=home)

    events: list[str] = []
    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.path = args[0] if args else None

        def __enter__(self) -> "DummyLock":
            events.append(f"lock_enter:{self.path}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(f"lock_exit:{self.path}")

    class DummyConnection:
        def __enter__(self) -> "DummyConnection":
            events.append("db_connection_enter")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append("db_connection_exit")

    class DummyServer:
        def __init__(self) -> None:
            self.should_exit = False

        def run(self) -> None:
            events.append("server_run")

    def fake_server_builder(*, app, host: str, port: int, reload: bool) -> DummyServer:
        events.append("server_builder")
        return DummyServer()

    dummy_connection = DummyConnection()

    def fake_open_db_connection(_path) -> DummyConnection:
        events.append("db_connection_open")
        return dummy_connection

    def fake_apply_migrations(connection) -> None:
        events.append("migrations_applied")
        assert connection == dummy_connection

    def fake_write_pid(path: Path, pid: int) -> None:
        events.append(f"pid_write:{path}:{pid}")

    def fake_remove_pid(path: Path) -> None:
        events.append(f"pid_remove:{path}")

    monkeypatch.setattr(cli_module, "ServiceLock", DummyLock)
    monkeypatch.setattr(cli_module, "_build_uvicorn_server", fake_server_builder)
    monkeypatch.setattr(cli_module, "_write_pid_file", fake_write_pid)
    monkeypatch.setattr(cli_module, "_remove_pid_file", fake_remove_pid)
    monkeypatch.setattr(cli_module, "open_sqlite_connection", fake_open_db_connection)

    def fake_apply(_: object) -> None:
        events.append("migrations_applied")
        assert _ == dummy_connection
    monkeypatch.setattr(cli_module.migrations, "apply_migrations", fake_apply)

    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host=None, port=None, reload=False)
    code = _command_serve(args)

    assert code == 0
    assert "db_connection_enter" in events
    assert "migrations_applied" in events
    assert events.index("migrations_applied") < events.index("server_builder")


def test_command_serve_fails_when_migrations_fail(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "service"
    initialize_local_layout(home=home)

    events: list[str] = []
    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.path = args[0] if args else None

        def __enter__(self) -> "DummyLock":
            events.append(f"lock_enter:{self.path}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(f"lock_exit:{self.path}")

    class DummyConnection:
        def __enter__(self) -> "DummyConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append("db_connection_exit")

    class DummyServer:
        def run(self) -> None:
            events.append("server_run")

    def fake_server_builder(*, app, host: str, port: int, reload: bool) -> DummyServer:
        events.append("server_builder")
        return DummyServer()

    def fake_open_db_connection(_path) -> DummyConnection:
        events.append("db_connection_open")
        return DummyConnection()

    def fake_apply_migrations(_conn) -> None:
        events.append("migrations_failed")
        raise RuntimeError("migration boom")

    monkeypatch.setattr(cli_module, "ServiceLock", DummyLock)
    monkeypatch.setattr(cli_module, "open_sqlite_connection", fake_open_db_connection)
    monkeypatch.setattr(cli_module.migrations, "apply_migrations", fake_apply_migrations)
    monkeypatch.setattr(cli_module, "_build_uvicorn_server", fake_server_builder)
    monkeypatch.setattr(cli_module, "_write_pid_file", lambda path, pid: events.append("pid_write"))
    monkeypatch.setattr(cli_module, "_remove_pid_file", lambda path: events.append("pid_remove"))

    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host=None, port=None, reload=False)
    code = _command_serve(args)

    assert code == 1
    assert "failed to apply migrations: migration boom" in events or "migrations_failed" in events
    assert "server_builder" not in events
    assert "server_run" not in events
