"""Canonical migration runner for Local-owned ``rounds.db``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import migrations as _shared

MIGRATION_DIR = Path(__file__).resolve().parent / "rounds_migrations"
LOCAL_SCHEMA_VERSION = 10
Migration = _shared.Migration
MigrationError = _shared.MigrationError
MigrationChecksumError = _shared.MigrationChecksumError
MigrationVersionError = _shared.MigrationVersionError
UnknownMigrationError = _shared.UnknownMigrationError


def load_migrations() -> tuple[Migration, ...]:
    return _shared.load_migrations(MIGRATION_DIR)


def apply_migrations(conn: sqlite3.Connection) -> tuple[Migration, ...]:
    return _shared.apply_migrations(conn, MIGRATION_DIR)
