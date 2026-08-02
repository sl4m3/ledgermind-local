from __future__ import annotations

from pathlib import Path

from v3_migration.mapper import map_record
from v3_migration.models import LegacyRecord


def _record_with_raw_markdown(raw_markdown: str) -> LegacyRecord:
    return LegacyRecord(fid="x.md", markdown_exists=True, metadata_exists=False, raw_markdown=raw_markdown)


def test_mapper_frontmatter_extraction_ignores_trailing_content():
    raw = "---\ntitle: T\ntarget: T\nphase: emergent\n---\nbody text"
    mapped = map_record(_record_with_raw_markdown(raw))
    assert mapped["title"] == "T"
    assert mapped["target"] == "T"
    assert mapped["phase"] == "EMERGENT"
    assert mapped["markdown_body"] == "body text"


def test_mapper_unknown_phase_defaults_to_pattern():
    mapped = map_record(_record_with_raw_markdown("---\nphase: unknown\n---"))
    assert mapped["phase"] == "PATTERN"


def test_mapper_unknown_vitality_defaults_to_active():
    mapped = map_record(_record_with_raw_markdown("---\nstatus: archived\n---"))
    assert mapped["vitality"] == "ACTIVE"


def test_mapper_handles_duplicate_colons_in_frontmatter():
    raw = "---\ntitle: T: subtitle\ntarget: T\n---\nbody"
    mapped = map_record(_record_with_raw_markdown(raw))
    assert mapped["title"] == "T: subtitle"


def test_mapper_normalizes_whitespace_in_fields():
    raw = "---\ntitle:   spaced   title  \ntarget:   target   \n---\nbody"
    mapped = map_record(_record_with_raw_markdown(raw))
    assert mapped["title"] == "spaced title"
    assert mapped["target"] == "target"


def test_mapper_uses_metadata_when_markdown_missing_fields():
    record = LegacyRecord(fid="x.md", markdown_exists=False, metadata_exists=True, metadata_row={"title": "meta", "target": "meta/t"})
    mapped = map_record(record)
    assert mapped["title"] == "meta"
    assert mapped["target"] == "meta/t"


def test_mapper_empty_raw_does_not_crash():
    record = LegacyRecord(fid="x.md", markdown_exists=True, metadata_exists=False, raw_markdown="")
    mapped = map_record(record)
    assert mapped["source_system"] == "legacy_import"
    assert mapped["phase"] == "PATTERN"
