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
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Settings
from ledgermind_local.bootstrap import (
    _build_markdown_root,
    _build_vector_store_root,
    _build_vectorizer_factory,
    build_projection_names,
    build_round_processing_worker,
    initialize_local_layout,
)
from ledgermind_local.config import LocalConfig
from ledgermind_local.diagnostics.integrity import run_database_integrity_checks
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.projections import (
    KnowledgeFTSProjection,
    KnowledgeMarkdownGitAuditProjection,
    KnowledgeMarkdownProjection,
    KnowledgeVectorProjection,
    ProjectionDispatcher,
    _ProjectionHandler,
)
from ledgermind_local.scheduler import OutboxWorker, ProcessingWorkerLoop
from ledgermind_local.service_lock import ServiceLock, ServiceLockError

OUTBOX_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


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


def _command_init(args: argparse.Namespace) -> int:
    rotate_token = bool(args.rotate_token)
    force = bool(args.force)
    if rotate_token and not force:
        print("--rotate-token requires --force")
        return 2

    paths, config, token = initialize_local_layout(
        home=Path(args.home).expanduser(),
        force=force,
        rotate_token=rotate_token,
    )
    print(f"initialized {paths.home}")
    print(f"config: {paths.config_file}")
    print(f"token file: {paths.token_file}")
    print(f"database: {paths.resolve_database_path(config.database_path)}")
    print(f"logs: {paths.logs_dir}")
    if not token:
        raise RuntimeError("token must be generated or loaded")
    return 0



def _command_serve(args: argparse.Namespace) -> int:
    paths, config, token = initialize_local_layout(home=Path(args.home).expanduser())
    database_path = paths.resolve_database_path(config.database_path)
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
                with open_sqlite_connection(database_path) as connection:
                    migrations.apply_migrations(connection)
                    if not _assert_database_invariants(connection):
                        return 1

                    projection_names = build_projection_names(config)
                    outbox_worker = _build_outbox_worker(
                        database_path=database_path,
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

                    processing_loop: ProcessingWorkerLoop | None = None
                    processing_thread: threading.Thread | None = None
                    if config.processing_enabled:
                        processing_worker = build_round_processing_worker(
                            database_path=database_path,
                            worker_id=f"processing:{os.getpid()}",
                            max_attempts=config.processing_max_attempts,
                            retry_delay_seconds=config.processing_retry_delay_seconds,
                        )
                        processing_loop = ProcessingWorkerLoop(
                            processing_worker,
                            poll_interval_seconds=config.processing_poll_interval_seconds,
                        )
                        processing_thread = threading.Thread(
                            target=processing_loop.run,
                            name="ledgermind-round-processing-worker",
                        )
                        processing_thread.start()

                    run_result = 1
                    _write_pid_file(paths.service_pid_file, os.getpid())
                    try:
                        settings = Settings(
                            database_path=database_path,
                            api_token=token,
                            service_lock_path=paths.service_lock_file,
                            max_raw_round_bytes=config.raw_round_max_bytes,
                            raw_round_retention_days=config.raw_round_retention_days,
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
                            on_terminate=lambda: _request_worker_stops(
                                outbox_worker=outbox_worker,
                                processing_loop=processing_loop,
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
                        if outbox_worker is not None:
                            outbox_worker.close()
                        processing_stopped = _stop_processing_worker(
                            loop=processing_loop,
                            worker_thread=processing_thread,
                        )
                        if not processing_stopped:
                            print("failed to stop round processing worker within timeout")
                            run_result = 1
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
    database_path: str | Path,
    projection_names: tuple[str, ...],
    config: LocalConfig,
    projection_poll_interval_seconds: float,
) -> OutboxWorker | None:
    projection_connection = open_sqlite_connection(database_path)
    try:
        handlers = _build_projection_handlers(
            connection=projection_connection,
            database_path=database_path,
            projection_names=projection_names,
            config=config,
        )
        if not handlers:
            _close_projection_connection(projection_connection)
            return None

        dispatcher = ProjectionDispatcher(handlers)
        return OutboxWorker(
            database_path=database_path,
            dispatcher=dispatcher,
            worker_id=f"serve:{os.getpid()}",
            poll_interval_seconds=projection_poll_interval_seconds,
            close_callback=lambda: _close_projection_connection(projection_connection),
        )
    except Exception:
        _close_projection_connection(projection_connection)
        raise


def _close_projection_handlers(handlers: Mapping[str, object]) -> None:
    for handler in handlers.values():
        close_handler = getattr(handler, "close", None)
        if callable(close_handler):
            close_handler()


def _close_projection_connection(connection: object) -> None:
    close_connection = getattr(connection, "close", None)
    if callable(close_connection):
        close_connection()


def _request_worker_stops(
    *,
    outbox_worker: OutboxWorker | None,
    processing_loop: ProcessingWorkerLoop | None,
) -> None:
    if outbox_worker is not None:
        outbox_worker.request_stop()
    if processing_loop is not None:
        processing_loop.request_stop()


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


def _stop_processing_worker(
    *,
    loop: ProcessingWorkerLoop | None,
    worker_thread: threading.Thread | None,
) -> bool:
    if loop is None or worker_thread is None:
        return True
    loop.request_stop()
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
        paths, config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    except Exception as exc:  # noqa: BLE001
        print(f"failed to resolve service paths: {exc}")
        return 1

    database_path = paths.resolve_database_path(config.database_path)
    if not database_path.exists():
        print(f"database file not found: {database_path}")
        return 1

    destination = args.destination
    if destination is None:
        destination = paths.backups_dir

    backup_base = _make_backup_base_path(destination, paths)
    backup_base.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_root = Path(staging)
        snapshot = staging_root / database_path.name
        manifest = staging_root / "backup_manifest.json"
        try:
            _create_sqlite_backup_copy(source=database_path, target=snapshot)
            manifest_payload = _build_backup_manifest(
                database_path=database_path,
                config_file=paths.config_file,
            )
            manifest.write_text(manifest_payload, encoding="utf-8")
            with zipfile.ZipFile(backup_base, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                _add_to_archive(archive, paths.config_file, paths.home)
                _add_to_archive(archive, paths.token_file, paths.home)
                _add_to_archive_with_name(archive, snapshot, database_path.name)
                _add_to_archive(archive, database_path.with_suffix(".markdown"), paths.home)
                _add_to_archive(archive, database_path.with_suffix(".vectors"), paths.home)
                _add_to_archive_with_name(archive, manifest, "backup_manifest.json")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to create backup archive: {exc}")
            return 1

    print(f"backup created: {backup_base}")
    return 0


def _command_backup_restore(args: argparse.Namespace) -> int:
    try:
        paths, config, _token = initialize_local_layout(home=Path(args.home).expanduser())
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
        else paths.resolve_database_path(config.database_path)
    )
    database_archive_name = paths.resolve_database_path(config.database_path).name

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
                        f"{paths.resolve_database_path(config.database_path).with_suffix('.markdown').name}",
                        f"{paths.resolve_database_path(config.database_path).with_suffix('.vectors').name}",
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
    database_path = paths.resolve_database_path(config.database_path)

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
