"""Tests for maintenance CLI commands."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path

from cli import main
from persistence import migrations, open_sqlite_connection


def test_rotate_token_updates_existing_token(tmp_path: Path) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0
    old_token = (home / "server.token").read_text(encoding="utf-8")

    assert main(["--home", str(home), "rotate-token"]) == 0

    new_token = (home / "server.token").read_text(encoding="utf-8")
    assert new_token != old_token


def test_rotate_token_reports_error_when_service_layout_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "service"

    import cli as cli_module

    def _fail(_: object = None, **_kwargs: object) -> None:
        raise RuntimeError("simulated initialize failure")

    monkeypatch.setattr(cli_module, "initialize_local_layout", _fail)
    assert main(["--home", str(home), "rotate-token"]) == 1


def test_backup_create_writes_zip_archive(tmp_path: Path) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0

    assert main(["--home", str(home), "backup", "create"]) == 0

    backup_dir = home / "backups"
    archives = sorted(backup_dir.glob("ledgermind-backup-*.zip"))
    assert archives
    with zipfile.ZipFile(archives[0], "r") as archive:
        names = set(archive.namelist())
    assert "ledgermind.db" in names
    assert "config.json" in names
    assert "server.token" in names


def test_backup_create_writes_to_requested_path(tmp_path: Path) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0

    target = home / "snapshot.zip"
    assert (
        main(["--home", str(home), "backup", "create", "--destination", str(target)])
        == 0
    )
    assert target.exists()


def test_backup_restore_reverts_database_to_backup(tmp_path: Path) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0

    database = home / "ledgermind.db"
    connection = open_sqlite_connection(database)
    try:
        migrations.apply_migrations(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_spaces (
                memory_space_id,
                display_name,
                source_client,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "space-a",
                "space A",
                "hermes",
                "2026-08-01T00:00:00+00:00",
                "2026-08-01T00:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    backup = home / "snapshot.zip"
    assert (
        main(
            ["--home", str(home), "backup", "create", "--destination", str(backup)]
        )
        == 0
    )

    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM knowledge_items")
        connection.execute("DELETE FROM memory_spaces")
        connection.commit()
    finally:
        connection.close()

    assert main(["--home", str(home), "backup", "restore", "--source", str(backup)]) == 0

    connection = sqlite3.connect(database)
    try:
        rows = list(
            connection.execute(
                "SELECT memory_space_id FROM memory_spaces ORDER BY memory_space_id"
            ).fetchall()
        )
    finally:
        connection.close()
    assert rows == [("space-a",)]


def test_backup_restore_fails_when_service_is_running(tmp_path: Path) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0

    backup = home / "snapshot.zip"
    assert main(["--home", str(home), "backup", "create", "--destination", str(backup)]) == 0

    lock_path = home / "service.lock"
    lock_path.write_text(
        json.dumps({"version": 1, "pid": os.getpid()}, sort_keys=True),
        encoding="utf-8",
    )

    assert (
        main(["--home", str(home), "backup", "restore", "--source", str(backup)]) == 1
    )


def test_migrate_v3_command_supports_dry_run_and_reports_unimplemented(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v3"
    source.mkdir()

    assert (
        main(["--home", str(tmp_path / "service"), "migrate-v3", "--source", str(source), "--dry-run"]) == 0
    ) is True
    assert (
        main(["--home", str(tmp_path / "service"), "migrate-v3", "--source", str(source)]) == 0
    ) is True
