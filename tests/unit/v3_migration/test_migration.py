from __future__ import annotations

from pathlib import Path

from v3_migration.models import LegacyRecord
from v3_migration.validator import validate_temp_database
from v3_migration.writer import write_temp_migration


def test_empty_source_produces_empty_temp_migration(tmp_path: Path) -> None:
    records = [LegacyRecord(fid="a.md", markdown_exists=True, metadata_exists=False, raw_markdown="---\ntitle: A\n---\nbody")]
    dest = tmp_path / "migration.db"
    manifests, _warnings = write_temp_migration(records=records, destination=dest, migration_id="m1", apply=True)
    assert len(manifests) == 1
    assert manifests[0].action == "migrated"
    valid, messages = validate_temp_database(dest)
    assert valid is True
    assert messages == []
