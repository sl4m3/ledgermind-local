from __future__ import annotations

import sqlite3
from pathlib import Path

from ledgermind_local.v3_migration.models import LegacyRecord
from ledgermind_local.v3_migration.validator import validate_temp_database
from ledgermind_local.v3_migration.writer import write_temp_migration


def test_empty_source_produces_empty_temp_migration(tmp_path: Path) -> None:
    records = [LegacyRecord(fid="a.md", markdown_exists=True, metadata_exists=False, raw_markdown="---\ntitle: A\n---\nbody")]
    dest = tmp_path / "migration.db"
    manifests, _warnings = write_temp_migration(records=records, destination=dest, migration_id="m1", apply=True)
    assert len(manifests) == 1
    assert manifests[0].action == "migrated"
    valid, messages = validate_temp_database(dest)
    assert valid is True
    assert messages == []
    with sqlite3.connect(dest) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "legacy_id_map" not in tables
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_dry_run_rolls_back_records_but_keeps_canonical_schema(tmp_path: Path) -> None:
    dest = tmp_path / "preview.db"
    record = LegacyRecord(
        fid="a.md",
        markdown_exists=True,
        metadata_exists=False,
        raw_markdown="---\ntitle: A\ntarget: t/a\n---\nbody",
    )
    manifests, _warnings = write_temp_migration(
        records=[record],
        destination=dest,
        migration_id="preview",
        apply=False,
    )
    assert manifests[0].action == "planned"
    assert validate_temp_database(dest) == (True, [])
    with sqlite3.connect(dest) as connection:
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_repeating_import_is_duplicate_in_canonical_source_round(tmp_path: Path) -> None:
    dest = tmp_path / "repeat.db"
    record = LegacyRecord(
        fid="a.md",
        markdown_exists=True,
        metadata_exists=False,
        raw_markdown="---\ntitle: A\ntarget: t/a\n---\nbody",
    )
    first, _ = write_temp_migration(
        records=[record], destination=dest, migration_id="first", apply=True
    )
    second, _ = write_temp_migration(
        records=[record], destination=dest, migration_id="second", apply=True
    )
    assert first[0].action == "migrated"
    assert second[0].action == "duplicate"
    with sqlite3.connect(dest) as connection:
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
