from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ledgermind_local.v3_migration.models import ConsistencyCategory, LegacyRecord


def _safe_fetchall(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(query).fetchall()
    except Exception:  # noqa: BLE001 - legacy schemas may not expose this table
        return []
    return [dict(row) for row in rows]


def _detect_meta_tables(connection: sqlite3.Connection) -> tuple[str | None, str | None]:
    tables = {
        row["name"]
        for row in _safe_fetchall(
            connection,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        )
    }
    table_candidates = [
        "semantic_meta",
        "knowledge_meta",
        "meta",
        "items",
        "knowledge",
        "decisions",
        "proposals",
    ]
    for table in table_candidates:
        if table in tables:
            return table, "*"
    return None, None


def _metadata_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = _safe_fetchall(connection, f"PRAGMA table_info({table})")
    return [row["name"] for row in rows]


def read_legacy_meta(source: Path) -> list[LegacyRecord]:
    records: list[LegacyRecord] = []
    if not source.exists():
        return records
    connection = sqlite3.connect(str(source))
    connection.row_factory = sqlite3.Row
    try:
        table, _ = _detect_meta_tables(connection)
        if table:
            columns = _metadata_columns(connection, table)
            column_map = {name.lower(): name for name in columns}
            query = ", ".join(f'"{column_map.get(name.lower(), name)}"' for name in columns) if columns else "*"
            rows = connection.execute(f"SELECT {query} FROM {table}").fetchall()
            for row in rows:
                data = dict(row)
                fid = str(
                    data.get("fid")
                    or data.get("id")
                    or data.get("file_id")
                    or data.get("path")
                    or ""
                )
                if not fid:
                    continue
                records.append(
                    LegacyRecord(
                        fid=fid,
                        markdown_exists=False,
                        metadata_exists=True,
                        metadata_row=data,
                    )
                )
    finally:
        connection.close()
    return records


def read_legacy_markdown(source: Path) -> dict[str, LegacyRecord]:
    mapping: dict[str, LegacyRecord] = {}
    if not source.exists() or not source.is_dir():
        return mapping
    for path in sorted(source.rglob("*.md")):
        fid = str(path.relative_to(source))
        raw = path.read_text(encoding="utf-8", errors="ignore")
        mapping[fid] = LegacyRecord(
            fid=fid,
            markdown_exists=True,
            metadata_exists=False,
            raw_markdown=raw,
        )
    return mapping


def merge_records(
    meta_records: list[LegacyRecord], markdown_map: dict[str, LegacyRecord]
) -> list[LegacyRecord]:
    merged: dict[str, LegacyRecord] = {}
    for record in meta_records:
        merged[record.fid] = record
    for fid, record in markdown_map.items():
        existing = merged.get(fid)
        if existing:
            existing.markdown_exists = True
            existing.raw_markdown = record.raw_markdown or existing.raw_markdown
            if existing.metadata_exists and existing.metadata_row and existing.raw_markdown:
                existing.category = ConsistencyCategory.CONSISTENT
            elif existing.metadata_exists:
                existing.category = ConsistencyCategory.META_ONLY
            else:
                existing.category = ConsistencyCategory.FILE_ONLY
        else:
            record.category = ConsistencyCategory.FILE_ONLY
            merged[fid] = record
    return list(merged.values())


def read_legacy_storage(source: Path) -> list[LegacyRecord]:
    if not source.exists():
        return []
    if source.is_file() and source.suffix.lower() == ".db":
        return read_legacy_meta(source)
    if source.is_dir():
        meta_files = [path for path in source.rglob("*.db")]
        if meta_files:
            return read_legacy_meta(meta_files[0])
        return list(read_legacy_markdown(source).values())
    return []
