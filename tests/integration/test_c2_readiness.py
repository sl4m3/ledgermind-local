from __future__ import annotations

from pathlib import Path

import pytest

from ledgermind_local.bootstrap import LocalRuntime
from ledgermind_local.config import LocalConfig
from ledgermind_local.core_gateway.contracts import (
    ControlMaintenanceResult,
    CoreHealth,
    ObjectFacetStatistics,
)
from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations


class _Gateway:
    def __init__(self, *, schema_version: int = 11) -> None:
        self.schema_version = schema_version

    def require_capabilities(self, *capabilities: str) -> None:
        del capabilities

    def health(self) -> CoreHealth:
        return CoreHealth(
            healthy=True,
            backend="fake",
            protocol_version=1,
            schema_version=self.schema_version,
        )

    def run_control_maintenance(self, command: object) -> ControlMaintenanceResult:
        del command
        return ControlMaintenanceResult(
            status="completed",
            object_count=0,
            active_value_count=0,
            superseded_value_count=0,
            operational_backlog=0,
            background_backlog=0,
            embedding_backlog=0,
            integrity_finding_count=0,
        )

    def get_object_facet_statistics(self, request_id: str) -> ObjectFacetStatistics:
        del request_id
        return ObjectFacetStatistics(
            object_count=0,
            active_value_count=0,
            superseded_value_count=0,
            operational_backlog=0,
            background_backlog=0,
            embedding_backlog=0,
            integrity_finding_count=0,
        )

    def close(self) -> None:
        return None


def _config() -> LocalConfig:
    return LocalConfig(
        config_version=1,
        workers={
            "retention": {"enabled": False},
            "processing": {"enabled": False},
            "core_commands": {"enabled": False},
            "core_projections": {"enabled": False},
            "core_model_tasks": {"enabled": True},
        },
    )


def _seed_profiles(database: Path, missing: str | None = None) -> None:
    database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(database)
    try:
        migrations.apply_migrations(connection)
        connection.execute(
            "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("space-c2", "tests", "2026-08-08T00:00:00Z", "2026-08-08T00:00:00Z"),
        )
        store = InferenceProfileStore(connection)
        for slot in ("operational", "background", "embedding"):
            profile_id = f"{slot}-profile"
            store.upsert(
                InferenceProfile(
                    profile_id=profile_id,
                    base_url="https://provider.example/v1",
                    model=f"{slot}-model",
                    secret_ref=f"{slot}-secret",
                )
            )
            if slot != missing:
                store.bind_slot("space-c2", slot=slot, profile_id=profile_id)
        connection.commit()
    finally:
        connection.close()


def _runtime(tmp_path: Path, *, gateway: _Gateway) -> LocalRuntime:
    paths = ServicePaths(tmp_path / "service")
    database = paths.resolve_rounds_database_path("rounds.db")

    class NoopWorker:
        def process_once(self) -> None:
            return None

        def close(self) -> None:
            return None

    return LocalRuntime(
        paths=paths,
        config=_config(),
        api_token="token",
        database_path=database,
        core_gateway_factory=lambda: gateway,
        worker_factories={"core_model_tasks": lambda _runtime: NoopWorker()},
    )


def test_full_readiness_requires_all_three_profile_slots(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, gateway=_Gateway())
    _seed_profiles(runtime.database_path)

    runtime.start()
    try:
        report = runtime.health_report()
        assert report["full_ready"] is True
        assert report["missing_profile_slots_by_memory_space"] == {}
    finally:
        runtime.stop()


@pytest.mark.parametrize("missing", ["operational", "background", "embedding"])
def test_full_readiness_reports_each_missing_profile_slot(
    tmp_path: Path, missing: str
) -> None:
    runtime = _runtime(tmp_path, gateway=_Gateway())
    _seed_profiles(runtime.database_path, missing=missing)

    runtime.start()
    try:
        report = runtime.health_report()
        assert report["full_ready"] is False
        assert report["missing_profile_slots_by_memory_space"] == {
            "space-c2": [missing]
        }
    finally:
        runtime.stop()


def test_full_readiness_requires_schema_eleven(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, gateway=_Gateway(schema_version=10))
    _seed_profiles(runtime.database_path)

    runtime.start()
    try:
        report = runtime.health_report()
        assert report["full_ready"] is False
        assert report["components"]["core"]["schema_version"] == 10
    finally:
        runtime.stop()
