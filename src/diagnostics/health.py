"""Health-related diagnostics for the local LedgerMind service."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from persistence import open_sqlite_connection
from persistence.migrations import (
    MigrationError,
    apply_migrations,
)

_DATABASE_ERROR = "database unavailable"


def _record(checks: dict[str, dict[str, Any]], name: str, ok: bool, reason: str) -> bool:
    checks[name] = {"ok": ok, "detail": reason}
    return ok


def _check_database_open(database_path: Path) -> tuple[bool, str]:
    try:
        connection = open_sqlite_connection(database_path)
        try:
            connection.execute("SELECT 1").fetchone()
        finally:
            connection.close()
        return True, "ok"
    except sqlite3.DatabaseError as exc:
        return False, f"{_DATABASE_ERROR}: {exc}"


def _check_migrations(database_path: Path) -> tuple[bool, str]:
    try:
        connection = open_sqlite_connection(database_path)
        try:
            apply_migrations(connection)
        finally:
            connection.close()
        return True, "ok"
    except (MigrationError, sqlite3.DatabaseError) as exc:
        return False, str(exc)


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


def _read_lock_payload(lock_path: Path) -> dict[str, Any]:
    raw = lock_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("service lock payload must be an object")
    return payload


def _check_service_lock(service_lock_path: Path | None) -> tuple[bool, str]:
    if service_lock_path is None:
        return True, "service lock path not configured"

    if not service_lock_path.exists():
        return False, "service lock file is missing"

    try:
        payload = _read_lock_payload(service_lock_path)
        owner_pid = int(payload.get("pid", 0))
    except Exception as exc:  # noqa: BLE001
        return False, f"service lock payload invalid: {exc}"

    if not _is_process_running(owner_pid):
        return False, "service lock owner process is not running"
    if owner_pid != os.getpid():
        return False, f"service lock is owned by pid {owner_pid}"

    return True, "ok"


def _check_write_handler(write_handler: object | None) -> tuple[bool, str]:
    if write_handler is None:
        return False, "write handler is not configured"
    handle = getattr(write_handler, "handle", None)
    if not callable(handle):
        return False, "write handler does not implement handle()"
    return True, "ok"


def run_readiness_checks(
    *,
    database_path: str | Path,
    service_lock_path: Path | None,
    write_handler: object | None,
) -> dict[str, object]:
    """Run local readiness checks and collect their status."""

    db_path = Path(database_path)
    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    database_open, database_detail = _check_database_open(db_path)
    if not _record(checks, "database_open", database_open, database_detail):
        issues.append(database_detail)
        return {
            "ready": False,
            "status": "unavailable",
            "checks": checks,
            "errors": issues,
        }

    migration_ok, migration_detail = _check_migrations(db_path)
    if not _record(checks, "migrations_applied", migration_ok, migration_detail):
        issues.append(migration_detail)

    lock_ok, lock_detail = _check_service_lock(service_lock_path)
    _record(checks, "service_lock", lock_ok, lock_detail)
    if not lock_ok and service_lock_path is not None:
        issues.append(lock_detail)

    write_ok, write_detail = _check_write_handler(write_handler)
    if not _record(checks, "write_handler", write_ok, write_detail):
        issues.append(write_detail)

    ready = len(issues) == 0
    return {
        "ready": ready,
        "status": "ok" if ready else "unavailable",
        "checks": checks,
        "errors": issues,
    }
