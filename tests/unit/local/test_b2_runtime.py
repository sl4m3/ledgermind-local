"""B2 runtime composition and lifecycle contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from ledgermind_local import cli
from ledgermind_local.bootstrap import LocalRuntime, _RuntimeCoreBackedSearch
from ledgermind_local.config import CoreSecurityConfig, LocalConfig
from ledgermind_local.core_gateway.contracts import (
    ContextViewResult,
    CoreHealth,
    RetrieveContextCommand,
)
from ledgermind_local.paths import ServicePaths
from ledgermind_local.search.core_backed import CandidateScore


class _Gateway:
    def __init__(self, events: list[str], *, healthy: bool = True) -> None:
        self.events = events
        self.healthy = healthy

    def require_capabilities(self, *capabilities: str) -> None:
        self.events.append("core-capabilities")

    def health(self) -> CoreHealth:
        self.events.append("core-health")
        return CoreHealth(healthy=self.healthy, backend="fake")

    def close(self) -> None:
        self.events.append("core-close")


class _Worker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def process_once(self) -> None:
        self.events.append("worker-iteration")

    def request_stop(self) -> None:
        self.events.append("worker-stop")


class _CandidateSearch:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(self, memory_space_id: str, query: str, limit: int) -> list[CandidateScore]:
        del memory_space_id, query
        self.limits.append(limit)
        return [CandidateScore("knowledge-1", 1.0, "fts")]


class _SearchGateway:
    def __init__(self) -> None:
        self.requests: list[RetrieveContextCommand] = []

    def retrieve_context(self, request: RetrieveContextCommand) -> ContextViewResult:
        self.requests.append(request)
        return ContextViewResult(())


@pytest.fixture
def runtime_config() -> LocalConfig:
    return LocalConfig(
        config_version=1,
        workers={
            "retention": {
                "enabled": True,
                "interval_seconds": 0.01,
                "shutdown_timeout_seconds": 1.0,
            },
            "core_commands": {"enabled": False},
            "core_projections": {"enabled": False},
            "core_model_tasks": {"enabled": False},
            "processing": {"enabled": False},
        },
    )


def _runtime(
    tmp_path: Path,
    config: LocalConfig,
    events: list[str],
    *,
    gateway: object | None = None,
    migration_runner=None,
    worker_factories=None,
) -> LocalRuntime:
    paths = ServicePaths(tmp_path / "service")
    database_path = paths.resolve_rounds_database_path(config.rounds_database_path)
    return LocalRuntime(
        paths=paths,
        config=config,
        api_token="test-token",
        core_gateway_factory=(lambda: gateway) if gateway is not None else None,
        migration_runner=migration_runner,
        worker_factories=worker_factories,
        database_path=database_path,
    )


def test_runtime_runs_migrations_before_any_worker_start(
    tmp_path: Path, runtime_config: LocalConfig
) -> None:
    events: list[str] = []

    def apply_migrations(connection: sqlite3.Connection) -> tuple[object, ...]:
        events.append("migrations")
        connection.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT)")
        connection.commit()
        return ()

    runtime = _runtime(
        tmp_path,
        runtime_config,
        events,
        gateway=_Gateway(events),
        migration_runner=apply_migrations,
        worker_factories={
            "retention": lambda _runtime: _Worker(events),
        },
    )

    runtime.start()
    try:
        assert events.index("migrations") < events.index("worker-iteration")
        assert runtime.capture_ready is True
        assert runtime.full_ready is True
        assert runtime.backup_service is not None
    finally:
        runtime.stop()


def test_runtime_failure_releases_lock_and_pid(tmp_path: Path, runtime_config: LocalConfig) -> None:
    def fail_migrations(_connection: sqlite3.Connection) -> tuple[object, ...]:
        raise RuntimeError("migration failure")

    runtime = _runtime(
        tmp_path,
        runtime_config,
        [],
        migration_runner=fail_migrations,
    )

    with pytest.raises(RuntimeError, match="migration failure"):
        runtime.start()

    assert not runtime.paths.service_lock_file.exists()
    assert not runtime.paths.service_pid_file.exists()
    assert runtime.capture_ready is False


def test_runtime_keeps_capture_ready_when_core_is_unavailable(
    tmp_path: Path, runtime_config: LocalConfig
) -> None:
    runtime = _runtime(tmp_path, runtime_config, [])

    def unavailable() -> object:
        raise RuntimeError("Core unavailable")

    runtime.core_gateway_factory = unavailable
    runtime.start()
    try:
        assert runtime.capture_ready is True
        assert runtime.full_ready is False
        report = runtime.health_report()
        assert report["capture_ready"] is True
        assert report["full_ready"] is False
        assert report["components"]["core"]["ready"] is False
    finally:
        runtime.stop()


def test_runtime_shutdown_stops_workers_before_core_and_releases_pid(
    tmp_path: Path, runtime_config: LocalConfig
) -> None:
    events: list[str] = []
    gateway = _Gateway(events)
    runtime = _runtime(
        tmp_path,
        runtime_config,
        events,
        gateway=gateway,
        worker_factories={
            "retention": lambda _runtime: _Worker(events),
        },
    )

    runtime.start()
    runtime.stop()

    assert "worker-stop" in events
    assert events.index("worker-stop") < events.index("core-close")
    assert not runtime.paths.service_lock_file.exists()
    assert not runtime.paths.service_pid_file.exists()


def test_secure_core_profile_requires_all_isolation_guarantees() -> None:
    config = LocalConfig(config_version=1)

    assert config.core_security.profile == "secure"
    assert config.core_security.require_network_isolation is True
    assert config.core_security.require_rounds_database_hidden is True
    assert config.core_security.require_filesystem_allowlist is True
    assert config.core_security.require_environment_sanitized is True
    assert config.core_security.require_signature is True


def test_secure_core_profile_rejects_disabled_guarantees() -> None:
    with pytest.raises(ValueError, match="secure profile requires"):
        LocalConfig(
            config_version=1,
            core_security={"profile": "secure", "require_signature": False},
        )


def test_permissive_core_profile_emits_runtime_warning(
    tmp_path: Path, runtime_config: LocalConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config = runtime_config.model_copy(
        update={
            "core_security": CoreSecurityConfig(profile="permissive"),
        }
    )
    runtime = _runtime(tmp_path, config, [])
    with caplog.at_level("WARNING", logger="ledgermind_local.bootstrap"):
        runtime.start()
    try:
        assert any("permissive" in record.message for record in caplog.records)
    finally:
        runtime.stop()


def test_cli_serve_secure_profile_refuses_unready_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = ServicePaths(tmp_path / "service")
    config = LocalConfig(config_version=1)
    runtime_state = {"stopped": False}

    class _UnreadyRuntime:
        full_ready = False

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            return None

        def health_report(self) -> dict[str, object]:
            return {
                "full_ready": False,
                "components": {"core": {"error_code": "CoreUnavailable"}},
            }

        def stop(self) -> None:
            runtime_state["stopped"] = True

        @property
        def database_path(self) -> Path:
            return paths.resolve_rounds_database_path(config.rounds_database_path)

    monkeypatch.setattr(
        cli,
        "initialize_local_layout",
        lambda *, home: (paths, config, "token"),
    )
    monkeypatch.setattr(cli, "LocalRuntime", _UnreadyRuntime)
    monkeypatch.setattr(
        cli,
        "_build_uvicorn_server",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("server started")),
    )

    code = cli._command_serve(  # type: ignore[arg-type]
        SimpleNamespace(home=str(tmp_path), host=None, port=None, reload=False)
    )

    assert code == 1
    assert runtime_state["stopped"] is True
    assert "refusing to serve" in capsys.readouterr().err


def test_search_candidate_multiplier_is_applied_per_request() -> None:
    local_search = _CandidateSearch()
    gateway = _SearchGateway()
    search = _RuntimeCoreBackedSearch(
        local_search=local_search,
        core_gateway=gateway,  # type: ignore[arg-type]
        candidate_multiplier=7,
        fallback_to_core_fts=True,
    )

    search.retrieve_context(
        RetrieveContextCommand(
            request_id="request-1",
            memory_space_id="space-1",
            query="query",
            limit=5,
        )
    )

    assert local_search.limits == [35]
    assert gateway.requests[0].candidate_ids == ("knowledge-1",)


def test_worker_config_exposes_all_b2_worker_names() -> None:
    config = LocalConfig(config_version=1)

    assert set(config.workers.model_dump()) == {
        "retention",
        "processing",
        "core_commands",
        "core_projections",
        "core_model_tasks",
    }
