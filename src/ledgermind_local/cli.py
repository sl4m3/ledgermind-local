"""Command-line interface for local service scaffolding."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Settings
from ledgermind_local.bootstrap import (
    build_core_projection_handlers,
    build_projection_names,
    build_round_processing_worker,
    initialize_local_layout,
)
from ledgermind_local.config import LocalConfig
from ledgermind_local.core_gateway import ProcessCoreGateway
from ledgermind_local.core_gateway.doctor import build_core_doctor_report
from ledgermind_local.core_gateway.signing import verify_core_binary
from ledgermind_local.core_gateway.supervisor import CoreSupervisor
from ledgermind_local.diagnostics.integrity import run_database_integrity_checks
from ledgermind_local.inference import (
    InferenceBroker,
    InferenceProfile,
    InferenceProfileStore,
    SecretStore,
)
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.scheduler import (
    CoreCommandWorker,
    CoreModelTaskWorker,
    CoreProjectionWorker,
    ProcessingWorkerLoop,
)
from ledgermind_local.service_lock import ServiceLock, ServiceLockError

WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _build_core_gateway(*, paths: ServicePaths, config: LocalConfig) -> Any:
    command_path = paths.resolve_core_path(config.core_binary_path)
    signature_path = paths.resolve_core_path(config.core_signature_path)
    public_key_path = paths.resolve_core_path(config.core_public_key_path)
    knowledge_database_path = paths.resolve_knowledge_database_path(
        config.knowledge_database_path
    )
    paths.core_data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if config.verify_core_signature:
        verify_core_binary(
            command_path,
            signature_path=signature_path,
            public_key_path=public_key_path,
        )
    supervisor = CoreSupervisor(
        [str(command_path), "--database", str(knowledge_database_path)],
        startup_timeout_seconds=config.core_startup_timeout_seconds,
        operation_timeout_seconds=config.core_request_timeout_seconds,
        core_data_dir=paths.core_data_dir,
        blocked_data_dirs=(paths.home,),
        require_network_isolation=config.require_core_network_isolation,
    )
    return ProcessCoreGateway(supervisor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LedgerMind local service")
    parser.add_argument(
        "--home",
        default="~/.ledgermind/local",
        help="Path to local data directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Initialize local storage directory"
    )
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

    status_parser = subparsers.add_parser(
        "status", help="Print configured service state"
    )
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
    doctor_parser.set_defaults(func=_command_doctor)

    core_parser = subparsers.add_parser("core", help="Rust Core operations")
    core_subparsers = core_parser.add_subparsers(
        dest="core_command",
        required=True,
    )
    core_doctor_parser = core_subparsers.add_parser(
        "doctor",
        help="Verify the Core binary and print secret-safe runtime diagnostics",
    )
    core_doctor_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable diagnostics",
    )
    core_doctor_parser.set_defaults(func=_command_core_doctor)

    rotate_token_parser = subparsers.add_parser(
        "rotate-token",
        help="Rotate local service API token in place",
    )
    rotate_token_parser.set_defaults(func=_command_rotate_token)

    backup_parser = subparsers.add_parser("backup", help="Backup operations")
    backup_subparsers = backup_parser.add_subparsers(
        dest="backup_command", required=True
    )
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

    profiles_parser = subparsers.add_parser(
        "profiles",
        help="Manage Local-owned inference profiles and memory-space bindings",
    )
    profiles_subparsers = profiles_parser.add_subparsers(
        dest="profiles_command",
        required=True,
    )

    profiles_list_parser = profiles_subparsers.add_parser(
        "list",
        help="List configured inference profiles",
    )
    profiles_list_parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only include enabled profiles",
    )
    profiles_list_parser.set_defaults(func=_command_profiles_list)

    profiles_add_parser = profiles_subparsers.add_parser(
        "add",
        help="Create or update an inference profile",
    )
    profiles_add_parser.add_argument("--id", dest="profile_id", required=True)
    profiles_add_parser.add_argument(
        "--provider",
        dest="provider_kind",
        choices=("openai_compatible",),
        default="openai_compatible",
    )
    profiles_add_parser.add_argument("--base-url", required=True)
    profiles_add_parser.add_argument("--model", required=True)
    profiles_add_parser.add_argument("--secret-ref", required=True)
    profiles_add_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    profiles_add_parser.add_argument("--max-retries", type=int, default=2)
    profiles_add_parser.add_argument("--max-input-tokens", type=int, default=12_000)
    profiles_add_parser.add_argument("--max-output-tokens", type=int, default=2_000)
    profiles_add_parser.add_argument("--disabled", action="store_true")
    profiles_add_parser.set_defaults(func=_command_profiles_add)

    profiles_remove_parser = profiles_subparsers.add_parser(
        "remove",
        help="Remove an inference profile",
    )
    profiles_remove_parser.add_argument("--id", dest="profile_id", required=True)
    profiles_remove_parser.set_defaults(func=_command_profiles_remove)

    profiles_bind_parser = profiles_subparsers.add_parser(
        "bind",
        help="Bind inference profiles to a memory space",
    )
    profiles_bind_parser.add_argument("--memory-space-id", required=True)
    profiles_bind_parser.add_argument("--hypothesis-profile")
    profiles_bind_parser.add_argument("--merge-profile")
    profiles_bind_parser.set_defaults(func=_command_profiles_bind)

    secrets_parser = subparsers.add_parser(
        "secrets",
        help="Manage Local-owned provider secrets without exposing values",
    )
    secrets_subparsers = secrets_parser.add_subparsers(
        dest="secrets_command",
        required=True,
    )

    secrets_set_parser = secrets_subparsers.add_parser(
        "set",
        help="Read one secret value from stdin and store it by reference",
    )
    secrets_set_parser.add_argument("--ref", dest="secret_ref", required=True)
    secrets_set_parser.add_argument(
        "--value-stdin",
        action="store_true",
        required=True,
        help="Read the secret value from stdin; never pass it as an argument",
    )
    secrets_set_parser.set_defaults(func=_command_secrets_set)

    secrets_list_parser = secrets_subparsers.add_parser(
        "list",
        help="List secret references without values",
    )
    secrets_list_parser.set_defaults(func=_command_secrets_list)

    secrets_delete_parser = secrets_subparsers.add_parser(
        "delete",
        help="Delete a secret by reference",
    )
    secrets_delete_parser.add_argument("--ref", dest="secret_ref", required=True)
    secrets_delete_parser.set_defaults(func=_command_secrets_delete)

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
    print(
        f"database: {paths.resolve_rounds_database_path(config.rounds_database_path)}"
    )
    print(f"logs: {paths.logs_dir}")
    if not token:
        raise RuntimeError("token must be generated or loaded")
    return 0


def _open_profile_store(
    args: argparse.Namespace,
) -> tuple[ServicePaths, sqlite3.Connection, InferenceProfileStore]:
    paths, config, _ = initialize_local_layout(home=Path(args.home).expanduser())
    database_path = paths.resolve_rounds_database_path(config.rounds_database_path)
    connection = open_sqlite_connection(database_path)
    migrations.apply_migrations(connection)
    return paths, connection, InferenceProfileStore(connection)


def _command_profiles_list(args: argparse.Namespace) -> int:
    _, connection, store = _open_profile_store(args)
    try:
        profiles = store.list(enabled_only=bool(args.enabled_only))
        print(
            json.dumps(
                [profile.model_dump(mode="json") for profile in profiles],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        connection.close()


def _command_profiles_add(args: argparse.Namespace) -> int:
    _, connection, store = _open_profile_store(args)
    try:
        profile = InferenceProfile(
            profile_id=args.profile_id,
            provider_kind=args.provider_kind,
            base_url=args.base_url,
            model=args.model,
            secret_ref=args.secret_ref,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            enabled=not bool(args.disabled),
        )
        store.upsert(profile)
        connection.commit()
        print(f"profile saved: {profile.profile_id} (secret_ref={profile.secret_ref})")
        return 0
    except ValueError as exc:
        connection.rollback()
        print(f"invalid profile: {exc}")
        return 2
    finally:
        connection.close()


def _command_profiles_remove(args: argparse.Namespace) -> int:
    _, connection, store = _open_profile_store(args)
    try:
        try:
            removed = store.remove(args.profile_id)
            if not removed:
                print(f"profile not found: {args.profile_id}")
                return 1
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            print("profile is still bound to a memory space")
            return 2
        print(f"profile removed: {args.profile_id}")
        return 0
    finally:
        connection.close()


def _command_profiles_bind(args: argparse.Namespace) -> int:
    if args.hypothesis_profile is None and args.merge_profile is None:
        print("at least one profile binding is required")
        return 2
    _, connection, store = _open_profile_store(args)
    try:
        try:
            store.bind(
                args.memory_space_id,
                hypothesis_profile_id=args.hypothesis_profile,
                merge_profile_id=args.merge_profile,
            )
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            print("invalid memory space or profile binding")
            return 2
        print(f"profiles bound: {args.memory_space_id}")
        return 0
    finally:
        connection.close()


def _open_secret_store(args: argparse.Namespace) -> SecretStore:
    paths, config, _ = initialize_local_layout(home=Path(args.home).expanduser())
    return SecretStore(paths.resolve_home_path(config.inference_secrets_path))


def _command_secrets_set(args: argparse.Namespace) -> int:
    store = _open_secret_store(args)
    value = sys.stdin.read().rstrip("\r\n")
    try:
        store.put(args.secret_ref, value)
    except (RuntimeError, ValueError) as exc:
        print(f"invalid secret: {exc}")
        return 2
    print(f"secret saved: {args.secret_ref}")
    return 0


def _command_secrets_list(args: argparse.Namespace) -> int:
    store = _open_secret_store(args)
    print(json.dumps(store.list_refs(), ensure_ascii=False))
    return 0


def _command_secrets_delete(args: argparse.Namespace) -> int:
    store = _open_secret_store(args)
    try:
        removed = store.delete(args.secret_ref)
    except (RuntimeError, ValueError) as exc:
        print(f"invalid secret reference: {exc}")
        return 2
    if not removed:
        print(f"secret not found: {args.secret_ref}")
        return 1
    print(f"secret deleted: {args.secret_ref}")
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    paths, config, token = initialize_local_layout(home=Path(args.home).expanduser())
    database_path = paths.resolve_rounds_database_path(config.rounds_database_path)
    bind_host = _coalesce_optional(args.host, config.bind_host)
    bind_port = _coalesce_optional(args.port, config.bind_port)
    if token is None:
        print("api token not configured")
        return 2
    if not _assert_bind_host_allowed(
        str(bind_host), allow_remote_bind=config.allow_remote_bind
    ):
        return 2

    try:
        with ServiceLock(paths.service_lock_file):
            try:
                with open_sqlite_connection(database_path) as connection:
                    migrations.apply_migrations(connection)
                    if not _assert_database_invariants(connection):
                        return 1

                    projection_names = build_projection_names(config)
                    processing_worker = None
                    core_command_worker = None
                    core_gateway = None
                    core_projection_worker = None
                    core_model_task_worker = None
                    if config.processing_enabled or config.core_backend == "process":
                        core_gateway = _build_core_gateway(paths=paths, config=config)
                    if config.processing_enabled:
                        if config.hypothesis_profile_id is None:
                            print("processing_enabled requires hypothesis_profile_id")
                            return 2
                        processing_worker = build_round_processing_worker(
                            database_path=database_path,
                            hypothesis_profile_id=config.hypothesis_profile_id,
                            secrets_path=paths.resolve_home_path(
                                config.inference_secrets_path
                            ),
                            worker_id=f"processing:{os.getpid()}",
                            max_attempts=config.processing_max_attempts,
                            retry_delay_seconds=config.processing_retry_delay_seconds,
                            lease_seconds=config.processing_lease_seconds,
                            heartbeat_interval_seconds=config.processing_heartbeat_interval_seconds,
                        )
                        assert core_gateway is not None
                        core_command_worker = CoreCommandWorker(
                            database_path=database_path,
                            gateway=core_gateway,
                            worker_id=f"core-command:{os.getpid()}",
                            max_attempts=config.processing_max_attempts,
                            retry_delay_seconds=config.processing_retry_delay_seconds,
                            lease_seconds=config.processing_lease_seconds,
                        )
                    if config.core_backend == "process":
                        assert core_gateway is not None
                        core_projection_worker = CoreProjectionWorker(
                            database_path=database_path,
                            gateway=core_gateway,
                            consumer_id="local-projections",
                            handlers_factory=lambda connection: build_core_projection_handlers(
                                connection=connection,
                                database_path=database_path,
                                config=config,
                            ),
                        )
                        core_model_task_worker = CoreModelTaskWorker(
                            database_path=database_path,
                            gateway=core_gateway,
                            broker=InferenceBroker(
                                database_path=database_path,
                                secret_store=SecretStore(
                                    paths.resolve_home_path(config.inference_secrets_path)
                                ),
                            ),
                            worker_id="local-model-tasks",
                            lease_seconds=int(config.processing_lease_seconds),
                        )
                    core_projection_loop: ProcessingWorkerLoop | None = None
                    core_projection_thread: threading.Thread | None = None
                    core_model_task_loop: ProcessingWorkerLoop | None = None
                    core_model_task_thread: threading.Thread | None = None
                    if core_projection_worker is not None:
                        core_projection_loop = ProcessingWorkerLoop(
                            core_projection_worker,
                            poll_interval_seconds=config.projection_poll_interval_seconds,
                        )
                        core_projection_thread = threading.Thread(
                            target=core_projection_loop.run,
                            name="ledgermind-core-projection-worker",
                        )
                        core_projection_thread.start()
                    if core_model_task_worker is not None:
                        core_model_task_loop = ProcessingWorkerLoop(
                            core_model_task_worker,
                            poll_interval_seconds=config.processing_poll_interval_seconds,
                        )
                        core_model_task_thread = threading.Thread(
                            target=core_model_task_loop.run,
                            name="ledgermind-core-model-task-worker",
                        )
                        core_model_task_thread.start()

                    processing_loop: ProcessingWorkerLoop | None = None
                    processing_thread: threading.Thread | None = None
                    core_command_loop: ProcessingWorkerLoop | None = None
                    core_command_thread: threading.Thread | None = None
                    if processing_worker is not None:
                        processing_loop = ProcessingWorkerLoop(
                            processing_worker,
                            poll_interval_seconds=config.processing_poll_interval_seconds,
                        )
                        processing_thread = threading.Thread(
                            target=processing_loop.run,
                            name="ledgermind-round-processing-worker",
                        )
                        processing_thread.start()
                    if core_command_worker is not None:
                        core_command_loop = ProcessingWorkerLoop(
                            core_command_worker,
                            poll_interval_seconds=config.processing_poll_interval_seconds,
                        )
                        core_command_thread = threading.Thread(
                            target=core_command_loop.run,
                            name="ledgermind-core-command-worker",
                        )
                        core_command_thread.start()

                    run_result = 1
                    _write_pid_file(paths.service_pid_file, os.getpid())
                    try:
                        settings = Settings(
                            rounds_database_path=database_path,
                            api_token=token,
                            service_lock_path=paths.service_lock_file,
                            max_raw_round_bytes=config.max_raw_round_bytes,
                            raw_round_retention_days=config.raw_round_retention_days,
                        )
                        app = create_app(
                            application=SimpleNamespace(core_gateway=core_gateway),
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
                                core_projection_loop=core_projection_loop,
                                core_model_task_loop=core_model_task_loop,
                                processing_loop=processing_loop,
                                core_command_loop=core_command_loop,
                            ),
                        )
                    finally:
                        core_projection_stopped = _stop_processing_worker(
                            loop=core_projection_loop,
                            worker_thread=core_projection_thread,
                        )
                        if not core_projection_stopped:
                            print("failed to stop Core projection worker within timeout")
                            run_result = 1
                        if core_projection_worker is not None:
                            core_projection_worker.close()
                        core_model_task_stopped = _stop_processing_worker(
                            loop=core_model_task_loop,
                            worker_thread=core_model_task_thread,
                        )
                        if not core_model_task_stopped:
                            print("failed to stop Core model task worker within timeout")
                            run_result = 1
                        if core_model_task_worker is not None:
                            core_model_task_worker.close()
                        processing_stopped = _stop_processing_worker(
                            loop=processing_loop,
                            worker_thread=processing_thread,
                        )
                        if not processing_stopped:
                            print(
                                "failed to stop round processing worker within timeout"
                            )
                            run_result = 1
                        core_command_stopped = _stop_processing_worker(
                            loop=core_command_loop,
                            worker_thread=core_command_thread,
                        )
                        if not core_command_stopped:
                            print("failed to stop Core command worker within timeout")
                            run_result = 1
                        if core_gateway is not None:
                            close_gateway = getattr(core_gateway, "close", None)
                            if callable(close_gateway):
                                close_gateway()
                        _checkpoint_wal_passive(connection)
                        _remove_pid_file(paths.service_pid_file)
                    return run_result
            except Exception as exc:  # noqa: BLE001
                print(f"failed to apply migrations: {exc}")
                return 1
    except ServiceLockError as exc:
        print(f"failed to start service: {exc}")
        return 1


def _request_worker_stops(
    *,
    core_projection_loop: ProcessingWorkerLoop | None,
    core_model_task_loop: ProcessingWorkerLoop | None,
    processing_loop: ProcessingWorkerLoop | None,
    core_command_loop: ProcessingWorkerLoop | None,
) -> None:
    if core_projection_loop is not None:
        core_projection_loop.request_stop()
    if core_model_task_loop is not None:
        core_model_task_loop.request_stop()
    if processing_loop is not None:
        processing_loop.request_stop()
    if core_command_loop is not None:
        core_command_loop.request_stop()


def _stop_processing_worker(
    *,
    loop: ProcessingWorkerLoop | None,
    worker_thread: threading.Thread | None,
) -> bool:
    if loop is None or worker_thread is None:
        return True
    loop.request_stop()
    worker_thread.join(WORKER_SHUTDOWN_TIMEOUT_SECONDS)
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
    app: Any,
    *,
    host: str,
    port: int,
    reload: bool,
) -> object:
    import uvicorn

    config = uvicorn.Config(
        app=app,
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
    previous: dict[
        int, Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
    ] = {}

    def _handle(_signal: int, _frame: FrameType | None) -> None:
        server.should_exit = True  # type: ignore[attr-defined]
        if on_terminate is not None:
            on_terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle)

    return previous


def _restore_signal_handlers(
    previous: dict[
        int, Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
    ],
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
        database_exists = paths.rounds_database_file.exists()
        config_exists = paths.config_file.exists()
        token_exists = paths.token_file.exists()
        print(f"service home: {paths.home}")
        print(
            f"database: {paths.rounds_database_file} ({'present' if database_exists else 'missing'})"
        )
        print(
            f"config: {paths.config_file} ({'present' if config_exists else 'missing'})"
        )
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
    database = (
        args.database if args.database is not None else paths.rounds_database_file
    )
    database = Path(database).expanduser()

    try:
        connection = open_sqlite_connection(database)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to open database for checks: {exc}")
        return 1

    try:
        issues = run_database_integrity_checks(connection)
    finally:
        connection.close()

    if issues:
        print("database integrity failed:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("database integrity checks passed")
    return 0


def _command_core_doctor(args: argparse.Namespace) -> int:
    paths = ServicePaths(home=Path(args.home).expanduser())
    try:
        if not paths.config_file.is_file():
            raise FileNotFoundError("Local config is missing")
        config = LocalConfig.from_file(paths.config_file)
        report = build_core_doctor_report(paths=paths, config=config)
    except Exception:  # noqa: BLE001
        report = {
            "ok": False,
            "error": "unable to load Core diagnostics configuration",
        }

    if bool(args.json_output):
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        _print_core_doctor_report(report)
    return 0 if report.get("ok") is True else 1


def _print_core_doctor_report(report: dict[str, Any]) -> None:
    print(f"core doctor: {'PASS' if report.get('ok') else 'FAIL'}")
    if "error" in report:
        print(f"error: {report['error']}")
        return
    binary = report["binary"]
    signature = report["signature"]
    sandbox = report["sandbox"]
    health = report["health"]
    environment = report["environment"]
    print(f"binary: {binary['path']} ({'present' if binary['present'] else 'missing'})")
    print(f"version: {report.get('version') or 'unavailable'}")
    print(f"sha256: {binary['sha256'] or 'unavailable'}")
    print(f"signature: {signature['status']}")
    print(f"sandbox: {sandbox['level']} ({sandbox['detail']})")
    print(
        "health: "
        f"{'healthy' if health['healthy'] else 'unhealthy'} "
        f"protocol={health['protocol_version'] or 'unavailable'} "
        f"schema={health['schema_version'] or 'unavailable'}"
    )
    print(f"environment keys: {', '.join(environment['keys']) or 'none'}")
    print(
        "secret-like environment keys: "
        f"{', '.join(environment['secret_like_keys']) or 'none'}"
    )


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
        paths, config, _token = initialize_local_layout(
            home=Path(args.home).expanduser()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"failed to resolve service paths: {exc}")
        return 1

    database_path = paths.resolve_rounds_database_path(config.rounds_database_path)
    knowledge_database_path = paths.resolve_knowledge_database_path(
        config.knowledge_database_path
    )
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
        knowledge_snapshot = staging_root / knowledge_database_path.name
        manifest = staging_root / "backup_manifest.json"
        try:
            _create_sqlite_backup_copy(source=database_path, target=snapshot)
            if knowledge_database_path.exists():
                shutil.copy2(knowledge_database_path, knowledge_snapshot)
            manifest_payload = _build_backup_manifest(
                database_path=database_path,
                knowledge_database_path=knowledge_database_path,
                config_file=paths.config_file,
            )
            manifest.write_text(manifest_payload, encoding="utf-8")
            with zipfile.ZipFile(
                backup_base, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                _add_to_archive(archive, paths.config_file, paths.home)
                _add_to_archive(archive, paths.token_file, paths.home)
                _add_to_archive_with_name(archive, snapshot, database_path.name)
                if knowledge_snapshot.exists():
                    _add_to_archive_with_name(
                        archive, knowledge_snapshot, knowledge_database_path.name
                    )
                _add_to_archive(
                    archive, database_path.with_suffix(".markdown"), paths.home
                )
                _add_to_archive(
                    archive, database_path.with_suffix(".vectors"), paths.home
                )
                _add_to_archive_with_name(archive, manifest, "backup_manifest.json")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to create backup archive: {exc}")
            return 1

    print(f"backup created: {backup_base}")
    return 0


def _command_backup_restore(args: argparse.Namespace) -> int:
    try:
        paths, config, _token = initialize_local_layout(
            home=Path(args.home).expanduser()
        )
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
        else paths.resolve_rounds_database_path(config.rounds_database_path)
    )
    target_knowledge_database = paths.resolve_knowledge_database_path(
        config.knowledge_database_path
    )
    database_archive_name = paths.resolve_rounds_database_path(
        config.rounds_database_path
    ).name
    knowledge_archive_name = target_knowledge_database.name

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
                        knowledge_archive_name,
                        "config.json",
                        "server.token",
                        "backup_manifest.json",
                        f"{paths.resolve_rounds_database_path(config.rounds_database_path).with_suffix('.markdown').name}",
                        f"{paths.resolve_rounds_database_path(config.rounds_database_path).with_suffix('.vectors').name}",
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
            target_knowledge_database=target_knowledge_database,
            knowledge_archive_name=knowledge_archive_name,
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
        print(
            f"restore requires stopped service; lock is held by running pid {owner_pid}"
        )
        return False

    service_lock_path.unlink(missing_ok=True)
    return True


def _restore_from_staging_path(
    *,
    staging_root: Path,
    target_paths: ServicePaths,
    target_database: Path,
    database_archive_name: str,
    target_knowledge_database: Path,
    knowledge_archive_name: str,
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

    staged_knowledge = staging_root / knowledge_archive_name
    if staged_knowledge.exists():
        temporary_knowledge = target_knowledge_database.with_name(
            f"{target_knowledge_database.name}.{os.getpid()}.restore-tmp"
        )
        try:
            target_knowledge_database.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_knowledge, temporary_knowledge)
            os.replace(temporary_knowledge, target_knowledge_database)
        except Exception as exc:  # noqa: BLE001
            temporary_knowledge.unlink(missing_ok=True)
            print(f"failed to restore Core knowledge artifact: {exc}")
            return False

    optional_targets = (
        ("config.json", target_paths.config_file),
        ("server.token", target_paths.token_file),
        (
            f"{target_paths.rounds_database_file.with_suffix('.markdown').name}",
            target_paths.home
            / f"{target_paths.rounds_database_file.with_suffix('.markdown').name}",
        ),
        (
            f"{target_paths.rounds_database_file.with_suffix('.vectors').name}",
            target_paths.home
            / f"{target_paths.rounds_database_file.with_suffix('.vectors').name}",
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


def _build_backup_manifest(
    *, database_path: Path, knowledge_database_path: Path, config_file: Path
) -> str:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(database_path.name),
        "knowledge_database": str(knowledge_database_path.name),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
