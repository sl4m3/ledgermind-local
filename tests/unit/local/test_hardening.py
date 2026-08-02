from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from cli import _command_migrate_v3, _build_parser
from v3_migration.writer import write_temp_migration
from v3_migration.models import LegacyRecord


def test_token_file_has_private_permissions_on_posix(tmp_path: Path) -> None:
    from cli import _command_init
    home = tmp_path / "service"
    home.mkdir()
    exit_code = _command_init(
        __import__("argparse").Namespace(home=str(home), force=False, rotate_token=False)
    )
    assert exit_code == 0
    token_file = home / "server.token"
    assert token_file.exists()
    mode = stat.S_IMODE(token_file.stat().st_mode)
    assert mode <= 0o640


def test_migrate_v3_requires_existing_source(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(["migrate-v3", "--source", str(tmp_path / "missing.db")])
    exit_code = _command_migrate_v3(args)
    assert exit_code == 2


def test_migrate_v3_dry_run_reports_without_writing_work_db(tmp_path: Path) -> None:
    import sqlite3

    source_db = tmp_path / "semantic_meta.db"
    with sqlite3.connect(source_db) as con:
        con.execute(
            "CREATE TABLE semantic_meta (fid TEXT PRIMARY KEY, title TEXT, target TEXT, status TEXT, namespace TEXT, context_json TEXT)"
        )
        con.execute(
            'INSERT INTO semantic_meta VALUES ("legacy.md","Legacy","legacy/t","active","default",?)',
            ('{"rationale":"safe"}',),
        )
        con.commit()

    work_db = tmp_path / "ledgermind.db"
    work_db.write_text("existing", encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args(["migrate-v3", "--source", str(source_db), "--dry-run"])
    exit_code = _command_migrate_v3(args)
    assert exit_code == 0
    assert work_db.read_text(encoding="utf-8") == "existing"


def test_temp_migration_uses_unique_destination(tmp_path: Path) -> None:
    records = [LegacyRecord(fid="a.md", markdown_exists=True, metadata_exists=False, raw_markdown="---\ntitle: A\n---\nbody")]
    dest = tmp_path / "migration.db"
    write_temp_migration(records=records, destination=dest, migration_id="m1", apply=True)
    assert dest.exists()
    assert not (tmp_path / "migration.db.tmp").exists()
