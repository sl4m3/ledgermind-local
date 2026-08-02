"""Command-line interface for local service scaffolding."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Settings
from ledgermind_local.bootstrap import (
    _build_markdown_root,
    _build_vector_store_root,
    _build_vectorizer_factory,
    build_projection_names,
    initialize_local_layout,
)
from ledgermind_local.config import LocalConfig
from ledgermind_local.diagnostics.integrity import run_database_integrity_checks
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.plugins.hermes.client import (
    LedgerMindClient,
    LedgerMindClientError,
)
from ledgermind_local.projections import (
    KnowledgeFTSProjection,
    KnowledgeMarkdownGitAuditProjection,
    KnowledgeMarkdownProjection,
    KnowledgeVectorProjection,
    ProjectionDispatcher,
    _ProjectionHandler,
)
from ledgermind_local.scheduler import OutboxWorker
from ledgermind_local.service_lock import ServiceLock, ServiceLockError

OUTBOX_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_HERMES_DEFAULT_HOME_ENV = "LEDGERMIND_HERMES_HOME"
_HERMES_PLUGIN_NAME = "ledgermind"
_HERMES_PLUGIN_DIR = "plugins"
_HERMES_DEFAULT_PROFILE = "default"


class _NoopProjectionHandler:
    def __init__(self, projection_name: str):
        self.projection_name = projection_name

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str | None = None,
    ) -> bool:
        del event_type
        del memory_space_id
        del aggregate_id
        del payload_json
        return False


def _resolve_hermes_home() -> Path:
    return Path(
        os.environ.get(
            _HERMES_DEFAULT_HOME_ENV,
            "~/.hermes",
        )
    ).expanduser()


def _normalize_profile_name(value: str | None) -> str:
    normalized = (value or "").strip()
    return normalized or _HERMES_DEFAULT_PROFILE


def _resolve_hermes_profile_home(*, profile: str, hermes_home: Path) -> Path:
    profile_name = _normalize_profile_name(profile)
    candidates = [
        hermes_home / "profiles" / profile_name,
        hermes_home / profile_name,
        hermes_home,
    ]
    for candidate in candidates[:-1]:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _read_profile_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    data = yaml.safe_load(raw)
    if isinstance(data, dict):
        return data
    return {}


def _write_profile_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _ensure_plugin_enabled_in_hermes_profile(profile_home: Path) -> None:
    candidates = [
        profile_home / "config.yaml",
        profile_home / "config.yml",
    ]
    config_path = next((path for path in candidates if path.exists()), candidates[0])
    payload = _read_profile_yaml(config_path)

    plugins = payload.get("plugins")
    if not isinstance(plugins, dict):
        plugins = {}
    enabled = plugins.get("enabled")
    if not isinstance(enabled, list):
        enabled = []
    if _HERMES_PLUGIN_NAME not in enabled:
        enabled.append(_HERMES_PLUGIN_NAME)

    plugins["enabled"] = enabled
    payload["plugins"] = plugins
    _write_profile_yaml(config_path, payload)
    print(f"updated hermes plugin list: {config_path}")


def _read_existing_plugin_config(plugin_config_path: Path) -> dict[str, object] | None:
    if not plugin_config_path.exists():
        return None
    raw = plugin_config_path.read_text(encoding="utf-8")
    if not raw.strip():
        return None
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return payload
    return None


def _write_plugin_config(
    plugin_config_path: Path,
    *,
    source_instance_id: str,
    service_url: str,
    token_file: Path,
    profile_name: str,
    memory_space_id: str,
    state_db_path: Path,
    extraction_prompt_version: int = 1,
    extraction_schema_version: int = 1,
) -> None:
    payload = {
        "config_version": 1,
        "source_instance_id": source_instance_id,
        "service_url": service_url,
        "token_file": str(token_file),
        "profile_name": profile_name,
        "memory_space_id": memory_space_id,
        "state_db_path": str(state_db_path),
        "extraction_prompt_version": extraction_prompt_version,
        "extraction_schema_version": extraction_schema_version,
        "pre_llm_timeout_seconds": 1.0,
        "delivery_timeout_seconds": 5.0,
        "max_context_items": 5,
    }
    plugin_config_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _copy_hermes_plugin_artifacts(*, source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if item.name == "__pycache__":
            continue
        if item.suffix == ".pyc":
            continue
        target = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, target)


def _build_service_url(config: LocalConfig) -> str:
    host = config.bind_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{config.bind_port}"


def _ensure_state_db_access(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("SELECT 1")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LedgerMind local service")
    parser.add_argument(
        "--home",
        default="~/.ledgermind/local",
        help="Path to local data directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize local storage directory")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config values",
    )
    init_parser.add_argument(
        "--rotate-token",
        action="store_true",
        help="Rotate token when force is used",
    )
    init_parser.set_defaults(func=_command_init)

    serve_parser = subparsers.add_parser("serve", help="Run local HTTP service")
    serve_parser.add_argument(
        "--host",
        help="Override bind host from config",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        help="Override bind port from config",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.set_defaults(func=_command_serve)

    status_parser = subparsers.add_parser("status", help="Print configured service state")
    status_parser.set_defaults(func=_command_status)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run local database integrity checks",
    )
    doctor_parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database file to check; defaults to ledgermind.db from --home",
    )
    doctor_parser.add_argument(
        "--allow-orphan-atoms",
        choices=("migration", "inactive"),
        help="Documented reason to allow atoms without evidence",
    )
    doctor_parser.set_defaults(func=_command_doctor)

    rotate_token_parser = subparsers.add_parser(
        "rotate-token",
        help="Rotate local service API token in place",
    )
    rotate_token_parser.set_defaults(func=_command_rotate_token)

    backup_parser = subparsers.add_parser("backup", help="Backup operations")
    backup_subparsers = backup_parser.add_subparsers(dest="backup_command", required=True)
    backup_create_parser = backup_subparsers.add_parser(
        "create",
        help="Create a local backup archive",
    )
    backup_create_parser.add_argument(
        "--destination",
        type=Path,
        help="Directory or file path for the backup archive (default: <home>/backups)",
    )
    backup_create_parser.set_defaults(func=_command_backup_create)

    backup_restore_parser = backup_subparsers.add_parser(
        "restore",
        help="Restore database and optional projections from backup",
    )
    backup_restore_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Backup archive path to restore",
    )
    backup_restore_parser.add_argument(
        "--database",
        type=Path,
        help="Target database path; defaults to layout database",
    )
    backup_restore_parser.set_defaults(func=_command_backup_restore)

    install_parser = subparsers.add_parser(
        "install",
        help="Install external integrations",
    )
    install_subparsers = install_parser.add_subparsers(dest="install_command", required=True)
    install_hermes_parser = install_subparsers.add_parser("hermes", help="Install Hermes plugin")
    install_hermes_parser.add_argument(
        "--profile",
        help="Hermes profile name (default: default)",
        default="default",
    )
    install_hermes_parser.set_defaults(func=_command_install_hermes)

    migrate_v3_parser = subparsers.add_parser(
        "migrate-v3",
        help="Migrate legacy v3 storage into v4 local store",
    )
    migrate_v3_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to legacy v3 source database/storage",
    )
    migrate_v3_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate migration source without changing current data",
    )
    migrate_v3_parser.add_argument(
        "--database",
        type=Path,
        help="Canonical target database; defaults to <home>/ledgermind.db",
    )
    migrate_v3_parser.set_defaults(func=_command_migrate_v3)

    rebuild_parser = subparsers.add_parser(
        "rebuild-projections",
        help="Rebuild derivative projections from canonical data",
    )
    rebuild_parser.add_argument(
        "--only",
        action="append",
        choices=("vector", "fts", "markdown", "markdown_audit"),
        help="Projection names to rebuild (vector, fts, markdown, markdown_audit)",
    )
    rebuild_parser.set_defaults(func=_command_rebuild_projections)

    return parser


def _command_migrate_v3(args: argparse.Namespace) -> int:
    source = args.source.expanduser().resolve()
    dry_run = bool(args.dry_run)
    if not source.exists():
        print(f"legacy source does not exist: {source}")
        return 2

    from ledgermind_local.v3_migration.reader import read_legacy_storage
    from ledgermind_local.v3_migration.validator import validate_temp_database
    from ledgermind_local.v3_migration.writer import write_temp_migration

    records = read_legacy_storage(source)
    migration_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = source.parent / f"ledgermind-v4-migration-{migration_id}.json"

    target_database = (
        args.database.expanduser().resolve()
        if args.database is not None
        else Path(args.home).expanduser().resolve() / "ledgermind.db"
    )
    temp_path = (
        source.parent / f"ledgermind-v4-migration-preview-{migration_id}.db"
        if dry_run
        else target_database
    )
    manifests, warnings = write_temp_migration(
        records=records,
        destination=temp_path,
        migration_id=migration_id,
        apply=not dry_run,
    )

    valid, validation_messages = validate_temp_database(temp_path)
    status = "dry_run" if dry_run else "applied"
    print(f"migration status={status} source={source}")
    print(f"source records={len(records)}")
    print(f"manifest records={len(manifests)}")
    print(f"warnings={len(warnings) + len(validation_messages)}")
    print(f"temp_database={temp_path}")
    print(f"manifest={manifest_path}")

    payload = [
        {
            "migration_id": item.migration_id,
            "legacy_fid": item.legacy_fid,
            "atom_id": item.atom_id,
            "knowledge_id": item.knowledge_id,
            "action": item.action,
            "warnings": json.loads(item.warnings_json),
        }
        for item in manifests
    ]
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not valid:
        for message in validation_messages:
            print(f"validation_error={message}")
        return 1
    return 0


def _command_init(args: argparse.Namespace) -> int:
    rotate_token = bool(args.rotate_token)
    force = bool(args.force)
    if rotate_token and not force:
        print("--rotate-token requires --force")
        return 2

    paths, _config, token = initialize_local_layout(
        home=Path(args.home).expanduser(),
        force=force,
        rotate_token=rotate_token,
    )
    print(f"initialized {paths.home}")
    print(f"config: {paths.config_file}")
    print(f"token file: {paths.token_file}")
    print(f"database: {paths.database_file}")
    print(f"logs: {paths.logs_dir}")
    if not token:
        raise RuntimeError("token must be generated or loaded")
    return 0


def _command_install_hermes(args: argparse.Namespace) -> int:
    profile = _normalize_profile_name(getattr(args, "profile", _HERMES_DEFAULT_PROFILE))
    profile_name = profile

    try:
        paths, service_config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    except Exception as exc:  # noqa: BLE001
        print(f"failed to initialize local service for install: {exc}")
        return 1

    if not _token:
        print("local service token is missing")
        return 1

    hermes_home = _resolve_hermes_home()
    profile_home = _resolve_hermes_profile_home(profile=profile_name, hermes_home=hermes_home)
    try:
        profile_home.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to prepare hermes profile directory: {exc}")
        return 1

    plugin_source_dir = Path(__file__).resolve().parent / _HERMES_PLUGIN_DIR / "hermes"
    plugin_destination = profile_home / _HERMES_PLUGIN_DIR / _HERMES_PLUGIN_NAME
    try:
        _copy_hermes_plugin_artifacts(
            source_dir=plugin_source_dir,
            destination_dir=plugin_destination,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to install plugin package: {exc}")
        return 1

    state_db_path = profile_home / "state.db"
    try:
        _ensure_state_db_access(state_db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to access hermes state database: {exc}")
        return 1

    plugin_config_path = plugin_destination / "config.json"
    existing_plugin_config = _read_existing_plugin_config(plugin_config_path)
    source_instance_id = (
        str(existing_plugin_config.get("source_instance_id"))
        if isinstance(existing_plugin_config, dict)
        and isinstance(existing_plugin_config.get("source_instance_id"), str)
        and existing_plugin_config.get("source_instance_id")
        else f"src_{uuid.uuid4().hex[:22]}"
    )
    memory_space_id = f"hermes:{source_instance_id}:{profile_name}"

    service_url = _build_service_url(service_config)
    try:
        _write_plugin_config(
            plugin_config_path,
            source_instance_id=source_instance_id,
            service_url=service_url,
            token_file=paths.token_file,
            profile_name=profile_name,
            memory_space_id=memory_space_id,
            state_db_path=state_db_path,
            extraction_prompt_version=1,
            extraction_schema_version=1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to write plugin configuration: {exc}")
        return 1

    try:
        _ensure_plugin_enabled_in_hermes_profile(profile_home)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to enable plugin in hermes profile config: {exc}")
        return 1

    try:
        client = LedgerMindClient(
            service_url=service_url,
            token_file=str(paths.token_file),
            timeout=2.0,
        )
        client.health(timeout=2.0)
        client.search_context(
            memory_space_id=memory_space_id,
            query="__ledgermind_plugin_install_probe__",
            limit=1,
            timeout=2.0,
        )
    except LedgerMindClientError as exc:
        print(f"local service is not reachable: {exc}")
        return 1

    print("hermes plugin installation complete")
    print(f"  hermes-home: {profile_home}")
    print(f"  plugin-dir: {plugin_destination}")
    print(f"  config: {plugin_config_path}")
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    paths, config, token = initialize_local_layout(home=Path(args.home).expanduser())
    bind_host = _coalesce_optional(args.host, config.bind_host)
    bind_port = _coalesce_optional(args.port, config.bind_port)
    if token is None:
        print("api token not configured")
        return 2
    if not _assert_bind_host_allowed(str(bind_host), allow_remote_bind=config.allow_remote_bind):
        return 2

    try:
        with ServiceLock(paths.service_lock_file):
            try:
                with open_sqlite_connection(paths.database_file) as connection:
                    migrations.apply_migrations(connection)
                    if not _assert_database_invariants(connection):
                        return 1

                    projection_names = build_projection_names(config)
                    handlers = _build_projection_handlers(
                        connection=connection,
                        database_path=paths.database_file,
                        projection_names=projection_names,
                        config=config,
                    )

                    outbox_worker = _build_outbox_worker(
                        connection=connection,
                        database_path=paths.database_file,
                        projection_names=projection_names,
                        config=config,
                        projection_poll_interval_seconds=config.projection_poll_interval_seconds,
                    )
                    outbox_thread: threading.Thread | None = None
                    if outbox_worker is not None:
                        outbox_thread = threading.Thread(
                            target=outbox_worker.run,
                            name="ledgermind-outbox-worker",
                        )
                        outbox_thread.start()

                    run_result = 1
                    _write_pid_file(paths.service_pid_file, os.getpid())
                    try:
                        settings = Settings(
                            database_path=paths.database_file,
                            api_token=token,
                            service_lock_path=paths.service_lock_file,
                        )
                        app = create_app(
                            application=object(),  # type: ignore[arg-type]
                            settings=settings,
                            projection_names=projection_names,
                        )
                        server = _build_uvicorn_server(
                            app=app,
                            host=bind_host,  # type: ignore[arg-type]
                            port=bind_port,  # type: ignore[arg-type]
                            reload=bool(getattr(args, "reload", False)),
                        )
                        run_result = _run_uvicorn_server(
                            server,
                            on_terminate=(
                                outbox_worker.request_stop if outbox_worker is not None else None
                            ),
                        )
                    finally:
                        worker_stopped = _stop_outbox_worker(
                            worker=outbox_worker,
                            worker_thread=outbox_thread,
                        )
                        if outbox_worker is not None and not worker_stopped:
                            print("failed to stop projection worker within timeout")
                            run_result = 1
                        _close_projection_handlers(handlers)
                        _checkpoint_wal_passive(connection)
                        _remove_pid_file(paths.service_pid_file)
                    return run_result
            except Exception as exc:  # noqa: BLE001
                print(f"failed to apply migrations: {exc}")
                return 1
    except ServiceLockError as exc:
        print(f"failed to start service: {exc}")
        return 1


def _build_projection_handlers(
    *,
    connection: object,
    database_path: str | Path,
    projection_names: tuple[str, ...],
    config: LocalConfig,
) -> dict[str, _ProjectionHandler]:
    if not hasattr(connection, "execute"):
        return {}

    handlers: dict[str, _ProjectionHandler] = {}
    database_path = Path(database_path)

    if "projections.search" in projection_names:
        handlers["projections.search"] = KnowledgeFTSProjection(connection=connection)  # type: ignore[arg-type]

    if "projections.knowledge" in projection_names:
        vectorizer_factory = _build_vectorizer_factory()
        if vectorizer_factory is None:
            handlers["projections.knowledge"] = _NoopProjectionHandler(
                projection_name="projections.knowledge",
            )
        else:
            handlers["projections.knowledge"] = KnowledgeVectorProjection(
                connection=connection,  # type: ignore[arg-type]
                vector_store_root=_build_vector_store_root(database_path),
                vectorizer_factory=vectorizer_factory,
            )

    if "projections.markdown" in projection_names:
        handlers["projections.markdown"] = KnowledgeMarkdownProjection(
            connection=connection,
            markdown_root=_build_markdown_root(database_path),
        )

    if "projections.markdown_audit" in projection_names:
        handlers["projections.markdown_audit"] = KnowledgeMarkdownGitAuditProjection(
            markdown_root=_build_markdown_root(database_path),
            enabled=bool(config.markdown_audit_enabled),
        )

    return handlers


def _build_outbox_worker(
    *,
    connection: object,
    database_path: str | Path,
    projection_names: tuple[str, ...],
    config: LocalConfig,
    projection_poll_interval_seconds: float,
) -> OutboxWorker | None:
    if not hasattr(connection, "execute"):
        return None

    handlers = _build_projection_handlers(
        connection=connection,
        database_path=database_path,
        projection_names=projection_names,
        config=config,
    )
    if not handlers:
        return None

    dispatcher = ProjectionDispatcher(handlers)
    return OutboxWorker(
        database_path=database_path,
        dispatcher=dispatcher,
        worker_id=f"serve:{os.getpid()}",
        poll_interval_seconds=projection_poll_interval_seconds,
        projection_handlers_factory=lambda uow: _build_projection_handlers(
            connection=uow.connection,
            database_path=database_path,
            projection_names=projection_names,
            config=config,
        ),
    )


def _close_projection_handlers(handlers: Mapping[str, object]) -> None:
    for handler in handlers.values():
        close_handler = getattr(handler, "close", None)
        if callable(close_handler):
            close_handler()


def _stop_outbox_worker(
    *,
    worker: OutboxWorker | None,
    worker_thread: threading.Thread | None,
) -> bool:
    if worker is None or worker_thread is None:
        return True

    worker.request_stop()
    worker_thread.join(OUTBOX_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
    return not worker_thread.is_alive()


def _checkpoint_wal_passive(connection: object) -> None:
    if not hasattr(connection, "execute"):
        return
    try:
        connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except Exception as exc:  # noqa: BLE001
        print(f"failed to checkpoint WAL passively: {exc}")


def _coalesce_optional(value: object, fallback: object) -> object:
    if value is None or value == "":
        return fallback
    return value


def _assert_database_invariants(connection: object) -> bool:
    if not hasattr(connection, "execute"):
        return True

    try:
        issues = run_database_integrity_checks(connection)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        print(f"failed to verify database invariants: {exc}")
        return False

    if issues:
        print("database invariants check failed:")
        for issue in issues:
            print(f" - {issue}")
        return False
    return True


def _is_local_bind_host(bind_host: str) -> bool:
    host = bind_host.strip().lower()
    return host in {"127.0.0.1", "localhost", "::1", "::", "::ffff:127.0.0.1"}


def _assert_bind_host_allowed(bind_host: str, *, allow_remote_bind: bool) -> bool:
    if _is_local_bind_host(bind_host):
        return True
    if not allow_remote_bind:
        print(
            "binding to non-localhost is disabled; set allow_remote_bind=true in config.json"
        )
        return False
    print(f"warning: binding to non-localhost host {bind_host!r} is enabled")
    return True


def _build_uvicorn_server(
    app: object,
    *,
    host: str,
    port: int,
    reload: bool,
) -> object:
    import uvicorn

    config = uvicorn.Config(
        app=app,  # type: ignore[arg-type]
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    return uvicorn.Server(config=config)


def _install_signal_handlers(
    server: object,
    *,
    on_terminate: Callable[[], None] | None = None,
) -> dict[int, Callable[[int, FrameType | None], Any] | int | signal.Handlers | None]:
    previous: dict[int, Callable[[int, FrameType | None], Any] | int | signal.Handlers | None] = {}

    def _handle(_signal: int, _frame: FrameType | None) -> None:
        server.should_exit = True  # type: ignore[attr-defined]
        if on_terminate is not None:
            on_terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle)

    return previous


def _restore_signal_handlers(
    previous: dict[int, Callable[[int, FrameType | None], Any] | int | signal.Handlers | None],
) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _run_uvicorn_server(
    server: object,
    *,
    on_terminate: Callable[[], None] | None = None,
) -> int:
    handlers = _install_signal_handlers(server, on_terminate=on_terminate)
    try:
        server.run()  # type: ignore[attr-defined]
        return 0
    finally:
        _restore_signal_handlers(handlers)


def _write_pid_file(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n", encoding="utf-8")
    os.chmod(pid_file, 0o600)


def _remove_pid_file(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


def _command_status(args: argparse.Namespace) -> int:
    try:
        paths = ServicePaths(home=Path(args.home).expanduser())
        database_exists = paths.database_file.exists()
        config_exists = paths.config_file.exists()
        token_exists = paths.token_file.exists()
        print(f"service home: {paths.home}")
        print(f"database: {paths.database_file} ({'present' if database_exists else 'missing'})")
        print(f"config: {paths.config_file} ({'present' if config_exists else 'missing'})")
        print(f"token: {paths.token_file} ({'present' if token_exists else 'missing'})")
        if not (database_exists and config_exists and token_exists):
            print("service layout is incomplete")
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"failed to read service status: {exc}")
        return 1


def _command_doctor(args: argparse.Namespace) -> int:
    home = Path(args.home).expanduser()
    paths = ServicePaths(home=home)
    database = args.database if args.database is not None else paths.database_file
    database = Path(database).expanduser()

    try:
        connection = open_sqlite_connection(database)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to open database for checks: {exc}")
        return 1

    try:
        issues = run_database_integrity_checks(
            connection,
            allow_orphan_atoms_reason=args.allow_orphan_atoms,
        )
    finally:
        connection.close()

    if issues:
        print("database integrity failed:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("database integrity checks passed")
    return 0


def _command_rotate_token(args: argparse.Namespace) -> int:
    try:
        paths, _config, token = initialize_local_layout(
            home=Path(args.home).expanduser(),
            force=False,
            rotate_token=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to rotate token: {exc}")
        return 1

    print(f"token rotated for {paths.home}")
    print(f"token file: {paths.token_file}")
    if not token:
        print("token is empty after rotation")
        return 1
    return 0


def _command_backup_create(args: argparse.Namespace) -> int:
    try:
        paths, _config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    except Exception as exc:  # noqa: BLE001
        print(f"failed to resolve service paths: {exc}")
        return 1

    if not paths.database_file.exists():
        print(f"database file not found: {paths.database_file}")
        return 1

    destination = args.destination
    if destination is None:
        destination = paths.backups_dir

    backup_base = _make_backup_base_path(destination, paths)
    backup_base.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_root = Path(staging)
        snapshot = staging_root / paths.database_file.name
        manifest = staging_root / "backup_manifest.json"
        try:
            _create_sqlite_backup_copy(source=paths.database_file, target=snapshot)
            manifest_payload = _build_backup_manifest(
                database_path=paths.database_file,
                config_file=paths.config_file,
            )
            manifest.write_text(manifest_payload, encoding="utf-8")
            with zipfile.ZipFile(backup_base, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                _add_to_archive(archive, paths.config_file, paths.home)
                _add_to_archive(archive, paths.token_file, paths.home)
                _add_to_archive_with_name(archive, snapshot, paths.database_file.name)
                _add_to_archive(archive, paths.home / paths.database_file.with_suffix(".markdown").name, paths.home)
                _add_to_archive(archive, paths.home / paths.database_file.with_suffix(".vectors").name, paths.home)
                _add_to_archive_with_name(archive, manifest, "backup_manifest.json")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to create backup archive: {exc}")
            return 1

    print(f"backup created: {backup_base}")
    return 0


def _command_backup_restore(args: argparse.Namespace) -> int:
    try:
        paths, _config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    except Exception as exc:  # noqa: BLE001
        print(f"failed to resolve service paths: {exc}")
        return 1

    source = Path(args.source).expanduser()
    if not source.is_file():
        print(f"backup source is missing: {source}")
        return 1

    target_database = (
        Path(args.database).expanduser()
        if args.database is not None
        else paths.database_file
    )
    database_archive_name = paths.database_file.name

    with tempfile.TemporaryDirectory() as staging:
        staging_root = Path(staging)
        try:
            with zipfile.ZipFile(source, "r") as archive:
                members = set(archive.namelist())
                if database_archive_name not in members:
                    print("backup does not contain a database file")
                    return 1

                selected = _collect_archive_members(
                    archive=archive,
                    member_prefixes=(
                        database_archive_name,
                        "config.json",
                        "server.token",
                        "backup_manifest.json",
                        f"{paths.database_file.with_suffix('.markdown').name}",
                        f"{paths.database_file.with_suffix('.vectors').name}",
                    ),
                    )
                for member in selected:
                    _safe_extract_zip_member(
                        archive=archive,
                        member=member,
                        destination=staging_root,
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"failed to extract backup archive: {exc}")
            return 1

        staged_db = staging_root / database_archive_name
        if not _validate_restoration_database(staged_db):
            return 1

        if not _assert_service_is_stopped(paths.service_lock_file):
            return 1

        if not _restore_from_staging_path(
            staging_root=staging_root,
            target_paths=paths,
            target_database=target_database,
            database_archive_name=database_archive_name,
        ):
            return 1

    print(f"restore completed: {target_database}")
    return 0


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _collect_archive_members(
    *,
    archive: zipfile.ZipFile,
    member_prefixes: tuple[str, ...],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    for entry in archive.namelist():
        if not _is_safe_archive_member(entry):
            continue
        for prefix in member_prefixes:
            if entry == prefix or entry.startswith(f"{prefix}/"):
                if entry not in seen:
                    seen.add(entry)
                    selected.append(entry)
                break

    return selected


def _is_safe_archive_member(member: str) -> bool:
    if not member:
        return False
    if member.startswith("/"):
        return False
    return ".." not in member.split("/")


def _safe_extract_zip_member(
    *,
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    if not _is_safe_archive_member(member):
        raise ValueError(f"unsafe archive member {member!r}")
    archive.extract(member, destination)


def _validate_restoration_database(staged_db: Path) -> bool:
    try:
        connection = open_sqlite_connection(staged_db)
    except Exception as exc:  # noqa: BLE001
        print(f"restoration validation failed to open staged database: {exc}")
        return False

    try:
        migrations.apply_migrations(connection)
        connection.commit()
        issues = run_database_integrity_checks(connection)
    except Exception as exc:  # noqa: BLE001
        print(f"restoration validation failed: {exc}")
        return False
    finally:
        connection.close()

    if issues:
        print("restoration validation failed:")
        for issue in issues:
            print(f" - {issue}")
        return False
    return True


def _assert_service_is_stopped(service_lock_path: Path) -> bool:
    if not service_lock_path.exists():
        return True

    payload = service_lock_path.read_text(encoding="utf-8")
    try:
        data = json.loads(payload)
    except Exception:  # noqa: BLE001
        service_lock_path.unlink(missing_ok=True)
        return True

    if not isinstance(data, dict):
        print("service lock payload is invalid")
        return False

    owner_pid = int(data.get("pid", 0))
    if _is_process_running(owner_pid):
        print(f"restore requires stopped service; lock is held by running pid {owner_pid}")
        return False

    service_lock_path.unlink(missing_ok=True)
    return True


def _restore_from_staging_path(
    *,
    staging_root: Path,
    target_paths: ServicePaths,
    target_database: Path,
    database_archive_name: str,
) -> bool:
    backup_database: Path | None = None
    staged_db = staging_root / database_archive_name
    if not staged_db.exists():
        print("staged database is missing after extraction")
        return False

    target_database = target_database.expanduser()
    try:
        if target_database.exists():
            backup_database = target_database.with_name(
                f"{target_database.name}.restore-old"
            )
            backup_database.unlink(missing_ok=True)
            os.replace(target_database, backup_database)

        target_root = target_database.parent
        target_root.mkdir(parents=True, exist_ok=True)
        temporary_target = target_database.with_name(
            f"{target_database.name}.{os.getpid()}.restore-tmp"
        )
        shutil.copy2(staged_db, temporary_target)
        os.replace(temporary_target, target_database)
    except Exception as exc:  # noqa: BLE001
        if backup_database is not None and backup_database.exists():
            try:
                os.replace(backup_database, target_database)
            except Exception as rollback_exc:  # noqa: BLE001
                print(f"restore rollback failed: {rollback_exc}")
        print(f"failed to restore database: {exc}")
        return False

    if backup_database is not None and backup_database.exists():
        backup_database.unlink(missing_ok=True)

    optional_targets = (
        ("config.json", target_paths.config_file),
        ("server.token", target_paths.token_file),
        (
            f"{target_paths.database_file.with_suffix('.markdown').name}",
            target_paths.home / f"{target_paths.database_file.with_suffix('.markdown').name}",
        ),
        (
            f"{target_paths.database_file.with_suffix('.vectors').name}",
            target_paths.home / f"{target_paths.database_file.with_suffix('.vectors').name}",
        ),
    )

    for member_name, destination in optional_targets:
        source = staging_root / member_name
        if not source.exists():
            continue
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    return True


def _create_sqlite_backup_copy(*, source: Path, target: Path) -> None:
    connection = sqlite3.connect(source)
    snapshot = sqlite3.connect(target)
    try:
        connection.backup(snapshot)
        snapshot.commit()
    finally:
        snapshot.close()
        connection.close()


def _build_backup_manifest(*, database_path: Path, config_file: Path) -> str:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(database_path.name),
        "config_path": str(config_file.name),
        "tool": "ledgermind-local-backup",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _make_backup_base_path(destination: Path, paths: ServicePaths) -> Path:
    destination = destination.expanduser()
    if destination.suffix and destination.name.lower().endswith(".zip"):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return destination / f"ledgermind-backup-{timestamp}.zip"


def _add_to_archive(archive: zipfile.ZipFile, source: Path, root: Path) -> None:
    if not source.exists():
        return
    if source.is_file():
        archive.write(source, arcname=str(source.relative_to(root)))
        return

    for file_path in source.rglob("*"):
        if file_path.is_file():
            archive.write(file_path, arcname=str(file_path.relative_to(root)))


def _add_to_archive_with_name(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
) -> None:
    archive.write(source, arcname=arcname)


def _command_rebuild_projections(args: argparse.Namespace) -> int:
    paths, config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    database_path = paths.database_file

    connection = open_sqlite_connection(database_path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"failed to ensure database migrations: {exc}")
        connection.close()
        return 1

    requested = set(getattr(args, "only", ()) or ("vector",))
    supported = {"vector", "fts", "markdown", "markdown_audit"}
    unknown = [item for item in requested if item not in supported]
    if unknown:
        print(f"unsupported projection(s) for rebuild: {', '.join(sorted(unknown))}")
        connection.close()
        return 2
    if "markdown_audit" in requested and not config.markdown_audit_enabled:
        print("markdown_audit projection is disabled in config")
        connection.close()
        return 2

    projections_to_rebuild: list[tuple[str, Any]] = []

    if "vector" in requested:
        vectorizer_factory = _build_vectorizer_factory()
        if vectorizer_factory is None:
            print("vector projections require LEDGERMIND_VECTOR_MODEL_PATH")
            connection.close()
            return 2
        projections_to_rebuild.append(
            (
                "vector",
                KnowledgeVectorProjection(
                    connection=connection,
                    vector_store_root=_build_vector_store_root(database_path),
                    vectorizer_factory=vectorizer_factory,
                ),
            )
        )

    if "fts" in requested:
        projections_to_rebuild.append(
            (
                "fts",
                KnowledgeFTSProjection(connection=connection),
            )
        )

    if "markdown" in requested:
        projections_to_rebuild.append(
            (
                "markdown",
                KnowledgeMarkdownProjection(
                    connection=connection,
                    markdown_root=_build_markdown_root(database_path),
                ),
            )
        )

    if "markdown_audit" in requested:
        projections_to_rebuild.append(
            (
                "markdown_audit",
                KnowledgeMarkdownGitAuditProjection(
                    markdown_root=_build_markdown_root(database_path),
                    enabled=True,
                ),
            )
        )

    results: list[str] = []
    success = False
    try:
        for projection_name, projection in projections_to_rebuild:
            rebuilt = projection.rebuild()
            results.append(f"{projection_name} projection rebuilt with {rebuilt} documents")
        connection.commit()
        success = True
    except Exception as exc:  # noqa: BLE001
        print(f"failed to rebuild projections: {exc}")
        try:
            connection.rollback()
        except Exception as rollback_exc:  # noqa: BLE001
            print(f"failed to rollback projection rebuild: {rollback_exc}")
        return 1
    finally:
        for _, projection in projections_to_rebuild:
            if hasattr(projection, "close"):
                projection.close()
        connection.close()

    if not success:
        return 1
    for message in results:
        print(message)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
