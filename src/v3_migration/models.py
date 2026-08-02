from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ConsistencyCategory(str, Enum):
    CONSISTENT = "CONSISTENT"
    FILE_ONLY = "FILE_ONLY"
    META_ONLY = "META_ONLY"
    NON_CRITICAL_MISMATCH = "NON_CRITICAL_MISMATCH"
    CRITICAL_MISMATCH = "CRITICAL_MISMATCH"
    UNPARSEABLE = "UNPARSEABLE"
    DANGLING_REFERENCE = "DANGLING_REFERENCE"
    CYCLE = "CYCLE"


@dataclass
class LegacyRecord:
    fid: str
    markdown_exists: bool
    metadata_exists: bool
    raw_markdown: str | None = None
    parsed_frontmatter: dict[str, Any] | None = None
    markdown_body: str | None = None
    metadata_row: dict[str, Any] | None = None
    git_history_summary: str | None = None
    category: ConsistencyCategory = ConsistencyCategory.CONSISTENT
    warnings: list[str] = field(default_factory=list)


@dataclass
class MigrationManifest:
    migration_id: str
    legacy_fid: str
    atom_id: str | None = None
    knowledge_id: str | None = None
    action: str | None = None
    warnings_json: str = "[]"
