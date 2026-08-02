from __future__ import annotations

import time
from pathlib import Path

from ledgermind_local.v3_migration.reader import read_legacy_storage
from ledgermind_local.v3_migration.validator import validate_temp_database
from ledgermind_local.v3_migration.writer import write_temp_migration


def _make_markdown_source(tmp_path: Path, count: int = 200) -> Path:
    source = tmp_path / "legacy_md"
    source.mkdir()
    for index in range(count):
        phase = "pattern" if index % 3 == 0 else "emergent" if index % 3 == 1 else "canonical"
        status = "active" if index % 5 != 0 else "deprecated"
        (source / f"{index}.md").write_text(
            f"---\ntitle: Item {index}\ntarget: t/{index}\nphase: {phase}\nstatus: {status}\n---\nbody {index}",
            encoding="utf-8",
        )
    return source


def test_soak_baseline_migrates_many_records_under_budget(tmp_path: Path) -> None:
    source = _make_markdown_source(tmp_path, count=200)
    dest = tmp_path / "migration.db"
    start = time.perf_counter()
    records = read_legacy_storage(source)
    manifests, warnings = write_temp_migration(records=records, destination=dest, migration_id="soak", apply=True)
    elapsed = time.perf_counter() - start
    valid, messages = validate_temp_database(dest)
    assert valid is True
    assert len(manifests) == 200
    assert warnings == []
    assert messages == []
    assert elapsed < 1.0


def test_soak_manifest_contains_mapped_fields_for_large_set(tmp_path: Path) -> None:
    source = _make_markdown_source(tmp_path, count=64)
    dest = tmp_path / "migration.db"
    records = read_legacy_storage(source)
    manifests, _ = write_temp_migration(records=records, destination=dest, migration_id="soak_manifest", apply=False)
    assert len(manifests) == 64
    for manifest in manifests:
        assert manifest.action == "planned"
        assert manifest.migration_id == "soak_manifest"
        assert manifest.atom_id is not None
        assert manifest.knowledge_id is not None
