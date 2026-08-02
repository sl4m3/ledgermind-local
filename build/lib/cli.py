"""Command-line interface for local service scaffolding."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
from typing import Sequence

from api.app import create_app
from api.dependencies import Settings
from bootstrap import initialize_local_layout
from service_lock import ServiceLock
from diagnostics.integrity import run_database_integrity_checks
from persistence import open_sqlite_connection
from paths import ServicePaths


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

    return parser


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


def _command_serve(args: argparse.Namespace) -> int:
    paths, config, token = initialize_local_layout(home=Path(args.home).expanduser())
    bind_host = _coalesce_optional(args.host, config.bind_host)
    bind_port = _coalesce_optional(args.port, config.bind_port)
    if token is None:
        print("api token not configured")
        return 2

    with ServiceLock(paths.service_lock_file):
        _write_pid_file(paths.service_pid_file, os.getpid())
        try:
            settings = Settings(
                database_path=paths.database_file,
                api_token=token,
                service_lock_path=paths.service_lock_file,
            )
            app = create_app(application=object(), settings=settings)
            server = _build_uvicorn_server(
                app=app,
                host=bind_host,
                port=bind_port,
                reload=bool(getattr(args, "reload", False)),
            )
            return _run_uvicorn_server(server)
        finally:
            _remove_pid_file(paths.service_pid_file)


def _coalesce_optional(value: object, fallback: object) -> object:
    if value is None or value == "":
        return fallback
    return value


def _build_uvicorn_server(
    app,
    *,
    host: str,
    port: int,
    reload: bool,
):
    import uvicorn

    return uvicorn.Server(
        uvicorn.Config(
            app=app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    )


def _install_signal_handlers(server) -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handle(_signal: int, _frame: object | None) -> None:
        server.should_exit = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle)

    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _run_uvicorn_server(server) -> int:
    handlers = _install_signal_handlers(server)
    try:
        server.run()
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
    paths, _config, _token = initialize_local_layout(home=Path(args.home).expanduser())
    print(f"service home: {paths.home}")
    print(f"database: {paths.database_file}")
    print(f"config: {paths.config_file}")
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    home = Path(args.home).expanduser()
    paths = ServicePaths(home=home)
    database = args.database if args.database is not None else paths.database_file
    database = Path(database).expanduser()

    connection = open_sqlite_connection(database)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
