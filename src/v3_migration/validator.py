from __future__ import annotations

import sqlite3
from pathlib import Path


def validate_temp_database(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, ["temp_database_missing"]
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    messages: list[str] = []
    try:
        fk_check = connection.execute("PRAGMA foreign_key_check").fetchall()
        if fk_check:
            messages.extend(f"foreign_key_violation:{row[0]}:{row[1]}:{row[2]}" for row in fk_check)

        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            messages.append(f"integrity_check_failed:{integrity[0] if integrity else 'no result'}")

        atoms = connection.execute("SELECT COUNT(*) AS count FROM atoms").fetchone()[0]
        knowledge_items = connection.execute("SELECT COUNT(*) AS count FROM knowledge_items").fetchone()[0]
        mapping_rows = connection.execute("SELECT COUNT(*) AS count FROM legacy_id_map").fetchone()[0]
        if mapping_rows == 0 and (atoms > 0 or knowledge_items > 0):
            messages.append("migration_map_empty")
        if atoms != knowledge_items:
            messages.append(f"atom_knowledge_count_mismatch:{atoms}:{knowledge_items}")
        if atoms != mapping_rows:
            messages.append(f"atom_mapping_count_mismatch:{atoms}:{mapping_rows}")
    finally:
        connection.close()
    return not messages, messages
