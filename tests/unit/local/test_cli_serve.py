"""Tests for local `ledgermind serve` command."""

from __future__ import annotations

import os
import signal
from argparse import Namespace
from pathlib import Path
from typing import Self

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import ledgermind_local.bootstrap as bootstrap_module
import ledgermind_local.cli as cli_module
from ledgermind_local.bootstrap import initialize_local_layout
from ledgermind_local.cli import (
    _build_core_gateway,
    _coalesce_optional,
    _command_serve,
    _install_signal_handlers,
    _restore_signal_handlers,
)
from ledgermind_local.config import LocalConfig
from ledgermind_local.core_gateway import ProcessCoreGateway
from ledgermind_local.core_gateway.signing import CoreBinaryVerificationError
from ledgermind_local.paths import ServicePaths
from ledgermind_local.service_lock import ServiceLockError


def _patch_noop_core_runtime(monkeypatch) -> None:
    class DummyGateway:
        def close(self) -> None:
            return None

    class DummyWorker:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def process_once(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        cli_module,
        "_build_core_gateway",
        lambda **kwargs: DummyGateway(),
    )
    monkeypatch.setattr(cli_module, "CoreProjectionWorker", DummyWorker)
    monkeypatch.setattr(bootstrap_module, "CoreExecutionTaskWorker", DummyWorker)


def test_coalesce_optional_returns_fallback() -> None:
    assert _coalesce_optional(None, "default") == "default"
    assert _coalesce_optional("", "default") == "default"
    assert _coalesce_optional(0, "default") == 0
    assert _coalesce_optional("0.0.0.0", "default") == "0.0.0.0"


def test_process_core_backend_builds_process_gateway_without_starting_core(
    tmp_path: Path,
) -> None:
    config = LocalConfig(
        config_version=1,
        core_backend="process",
        core_binary_path="bin/fake-core",
        verify_core_signature=False,
    )

    gateway = _build_core_gateway(paths=ServicePaths(tmp_path), config=config)

    assert isinstance(gateway, ProcessCoreGateway)


def test_process_core_backend_passes_core_owned_database_to_daemon(
    tmp_path: Path,
) -> None:
    config = LocalConfig(
        config_version=1,
        core_backend="process",
        core_binary_path="bin/ledgermind-core",
        knowledge_database_path="knowledge.db",
        verify_core_signature=False,
    )

    paths = ServicePaths(tmp_path)
    gateway = _build_core_gateway(paths=paths, config=config)

    assert gateway._supervisor._command == (
        str(paths.core_data_dir / "bin" / "ledgermind-core"),
        "--database",
        str(paths.core_data_dir / "knowledge.db"),
    )


def test_process_core_backend_verifies_signed_binary_before_gateway_creation(
    tmp_path: Path,
) -> None:
    paths = ServicePaths(tmp_path)
    binary = paths.core_data_dir / "bin" / "ledgermind-core"
    signature = paths.core_data_dir / "bin" / "ledgermind-core.sig"
    public_key = paths.core_data_dir / "bin" / "ledgermind-core.pub"
    binary.parent.mkdir(mode=0o700, parents=True)
    binary_bytes = b"signed daemon fixture"
    binary.write_bytes(binary_bytes)
    private_key = Ed25519PrivateKey.generate()
    signature.write_bytes(private_key.sign(binary_bytes))
    public_key.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )
    config = LocalConfig(
        config_version=1,
        core_backend="process",
        core_binary_path="bin/ledgermind-core",
        core_signature_path="bin/ledgermind-core.sig",
        core_public_key_path="bin/ledgermind-core.pub",
        verify_core_signature=True,
    )

    gateway = _build_core_gateway(paths=paths, config=config)
    assert isinstance(gateway, ProcessCoreGateway)
    gateway.close()

    binary.write_bytes(b"tampered daemon fixture")
    with pytest.raises(CoreBinaryVerificationError):
        _build_core_gateway(paths=paths, config=config)


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
        config=LocalConfig(
            config_version=1, bind_host="127.0.0.1", allow_remote_bind=False
        ),
    )

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("lock should never be created for rejected bind host")

        def __enter__(self) -> Self:
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

        def __enter__(self) -> Self:
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
    _patch_noop_core_runtime(monkeypatch)
    home = tmp_path / "service"
    initialize_local_layout(
        home=home,
        config=LocalConfig(
            config_version=1, allow_remote_bind=True, bind_host="127.0.0.1"
        ),
    )

    events: list[str] = []

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append(f"lock_enter:{args[0]}")

        def __enter__(self) -> Self:
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
    monkeypatch.setattr(
        cli_module, "_write_pid_file", lambda path, pid: events.append("pid_write")
    )
    monkeypatch.setattr(
        cli_module, "_remove_pid_file", lambda path: events.append("pid_remove")
    )
    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host="0.0.0.0", port=None, reload=False)
    code = _command_serve(args)

    assert code == 0
    assert "server_builder:0.0.0.0:8765:False" in events
    assert events[0].startswith("lock_enter:")
    assert "server_run" in events


def test_command_serve_writes_pid_and_starts_server(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_noop_core_runtime(monkeypatch)
    home = tmp_path / "service"
    initialize_local_layout(home=home)
    token = (home / "server.token").read_text(encoding="utf-8").strip()
    assert token

    events: list[str] = []

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.path = args[0] if args else None

        def __enter__(self) -> Self:
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


def test_command_serve_applies_migrations_before_starting_server(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_noop_core_runtime(monkeypatch)
    home = tmp_path / "service"
    initialize_local_layout(home=home)

    events: list[str] = []

    class DummyLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.path = args[0] if args else None

        def __enter__(self) -> Self:
            events.append(f"lock_enter:{self.path}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(f"lock_exit:{self.path}")

    class DummyConnection:
        def __enter__(self) -> Self:
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

        def __enter__(self) -> Self:
            events.append(f"lock_enter:{self.path}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            events.append(f"lock_exit:{self.path}")

    class DummyConnection:
        def __enter__(self) -> Self:
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
    monkeypatch.setattr(
        cli_module.migrations, "apply_migrations", fake_apply_migrations
    )
    monkeypatch.setattr(cli_module, "_build_uvicorn_server", fake_server_builder)
    monkeypatch.setattr(
        cli_module, "_write_pid_file", lambda path, pid: events.append("pid_write")
    )
    monkeypatch.setattr(
        cli_module, "_remove_pid_file", lambda path: events.append("pid_remove")
    )

    monkeypatch.setattr(os, "getpid", lambda: 12345)

    args = Namespace(home=str(home), host=None, port=None, reload=False)
    code = _command_serve(args)

    assert code == 1
    assert (
        "failed to apply migrations: migration boom" in events
        or "migrations_failed" in events
    )
    assert "server_builder" not in events
    assert "server_run" not in events
