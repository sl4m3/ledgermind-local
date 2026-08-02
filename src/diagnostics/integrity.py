"""Canonical SQLite integrity diagnostics for the local LedgerMind database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

_MAX_ISSUE_ITEMS = 10
_MAX_CYCLE_LENGTH = 32


def _preview(values: Iterable[str], *, limit: int = _MAX_ISSUE_ITEMS) -> str:
    items = list(values)
    if len(items) <= limit:
        return ", ".join(items)

    preview = ", ".join(items[:limit])
    return f"{preview}, ... (+{len(items) - limit} more)"


def _check_integrity_pragma(connection: sqlite3.Connection) -> list[str]:
    rows = list(connection.execute("PRAGMA integrity_check").fetchall())
    if len(rows) != 1 or str(rows[0][0]) != "ok":
        rows_repr = ", ".join(str(row[0]) for row in rows)
        return [f"integrity_check failed: {rows_repr}"]
    return []


def _check_foreign_key_pragma(connection: sqlite3.Connection) -> list[str]:
    rows = list(connection.execute("PRAGMA foreign_key_check").fetchall())
    if not rows:
        return []

    issues = []
    for row in rows:
        table = row[0]
        row_id = row[1]
        parent_table = row[2]
        details = row[3]
        issues.append(
            f"{table}.{row_id}: missing parent {parent_table}({details})"
        )

    return [f"foreign_key_check failed: {_preview(issues)}"]


def _check_orphaned_atoms(
    connection: sqlite3.Connection,
    *,
    allow_orphan_atoms_reason: str | None,
) -> list[str]:
    rows = list(
        connection.execute(
            """
            SELECT a.atom_id
            FROM atoms a
            LEFT JOIN knowledge_evidence e ON e.atom_id = a.atom_id
            WHERE e.knowledge_id IS NULL
            ORDER BY a.atom_id
            """
        ).fetchall()
    )
    if not rows:
        return []
    if allow_orphan_atoms_reason:
        return []

    atom_ids = [str(row[0]) for row in rows]
    return [
        (
            "orphan atoms detected: atoms without knowledge links are allowed only with "
            "--allow-orphan-atoms={migration|inactive}. "
            f"Found: {_preview(atom_ids)}"
        ),
    ]


def _check_current_knowledge_has_evidence(
    connection: sqlite3.Connection,
) -> list[str]:
    rows = list(
        connection.execute(
            """
            SELECT
                k.knowledge_id,
                COUNT(e.atom_id) AS evidence_count
            FROM knowledge_items k
            LEFT JOIN knowledge_evidence e ON e.knowledge_id = k.knowledge_id
            WHERE k.deleted_at IS NULL
              AND k.superseded_by_id IS NULL
            GROUP BY k.knowledge_id
            HAVING evidence_count = 0
            ORDER BY k.knowledge_id
            """
        ).fetchall()
    )
    if not rows:
        return []

    knowledge_ids = [row[0] for row in rows]
    return [
        f"current knowledge missing evidence: {_preview([str(value) for value in knowledge_ids])}"
    ]


def _check_knowledge_revision_versions(connection: sqlite3.Connection) -> list[str]:
    rows = list(
        connection.execute(
            """
            SELECT
                k.knowledge_id,
                k.version AS knowledge_version,
                MAX(r.version) AS revision_version
            FROM knowledge_items k
            LEFT JOIN knowledge_revisions r
                ON r.knowledge_id = k.knowledge_id
            GROUP BY k.knowledge_id
            HAVING revision_version IS NULL OR revision_version <> k.version
            ORDER BY k.knowledge_id
            """
        ).fetchall()
    )
    if not rows:
        return []

    details = []
    for row in rows:
        details.append(
            f"{row[0]} (knowledge={row[1]}, revision={row[2]})"
        )
    return [f"knowledge version mismatch: {_preview(details)}"]


def _check_superseded_cycles(connection: sqlite3.Connection) -> list[str]:
    rows = list(
        connection.execute(
            """
            SELECT knowledge_id, superseded_by_id
            FROM knowledge_items
            WHERE superseded_by_id IS NOT NULL
            """
        ).fetchall()
    )
    graph = {str(row[0]): str(row[1]) for row in rows if row[1] is not None}

    if not graph:
        return []

    state: dict[str, int] = {}
    index: dict[str, int] = {}
    visiting: list[str] = []
    cycles: list[tuple[str, ...]] = []

    def _walk(node: str) -> None:
        state[node] = 1
        index[node] = len(visiting)
        visiting.append(node)

        next_node = graph.get(node)
        if next_node in graph:
            next_state = state.get(next_node, 0)
            if next_state == 0:
                _walk(next_node)
            elif next_state == 1:
                cycle_start = index[next_node]
                cycle = tuple(visiting[cycle_start:] + [next_node])
                if len(cycle) <= _MAX_CYCLE_LENGTH:
                    cycles.append(cycle)
                else:
                    cycles.append(tuple(visiting[cycle_start:cycle_start+_MAX_CYCLE_LENGTH]))

        visiting.pop()
        index.pop(node, None)
        state[node] = 2

    for node in graph:
        if state.get(node, 0) != 0:
            continue
        _walk(node)

    deduped = []
    seen = set()
    for cycle in cycles:
        canonical = tuple(cycle)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(cycle)

    cycle_texts = [" -> ".join(cycle) for cycle in deduped]
    return [f"superseded_by_id cycle: {_preview(cycle_texts)}"]


def _check_evidence_memory_space_alignment(connection: sqlite3.Connection) -> list[str]:
    rows = list(
        connection.execute(
            """
            SELECT
                e.knowledge_id,
                e.atom_id,
                k.memory_space_id AS knowledge_space,
                a.memory_space_id AS atom_space
            FROM knowledge_evidence e
            JOIN knowledge_items k ON k.knowledge_id = e.knowledge_id
            JOIN atoms a ON a.atom_id = e.atom_id
            WHERE k.memory_space_id <> a.memory_space_id
            ORDER BY k.memory_space_id, a.memory_space_id, e.knowledge_id, e.atom_id
            """
        ).fetchall()
    )
    if not rows:
        return []

    details = []
    for row in rows:
        details.append(
            f"{row[0]}:{row[1]} ({row[2]} != {row[3]})"
        )
    return [
        f"evidence crosses memory spaces: {_preview(details)}"
    ]


def run_database_integrity_checks(
    connection: sqlite3.Connection,
    *,
    allow_orphan_atoms_reason: str | None = None,
) -> list[str]:
    """Run all database integrity checks and return textual findings."""

    issues: list[str] = []
    issues.extend(_check_integrity_pragma(connection))
    issues.extend(_check_foreign_key_pragma(connection))
    issues.extend(
        _check_orphaned_atoms(
            connection,
            allow_orphan_atoms_reason=allow_orphan_atoms_reason,
        )
    )
    issues.extend(_check_current_knowledge_has_evidence(connection))
    issues.extend(_check_knowledge_revision_versions(connection))
    issues.extend(_check_superseded_cycles(connection))
    issues.extend(_check_evidence_memory_space_alignment(connection))
    return issues


__all__ = ["run_database_integrity_checks"]
