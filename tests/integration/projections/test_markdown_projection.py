"""Integration tests for local markdown knowledge projection."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from domain.events import KnowledgeCreated, KnowledgeDeleted, KnowledgeSuperseded
from domain import EvidenceRelation
from persistence import (
    Knowledge,
    KnowledgeEvidence,
    SQLiteKnowledgeRepository,
    SQLiteEvidenceRepository,
    migrations,
    open_sqlite_connection,
)
from projections.markdown import KnowledgeMarkdownProjection


def _bootstrap(path: Path) -> None:
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _build_connection(path: Path):
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _ensure_space(connection, memory_space_id: str) -> None:
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
            memory_space_id,
            None,
            "hermes",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )


def _add_knowledge(
    connection,
    *,
    memory_space_id: str,
    knowledge_id: str,
    title: str,
    target: str,
    statement: str,
    rationale: str = "",
    superseded_by_id: str | None = None,
    deleted_at: str | None = None,
) -> None:
    SQLiteKnowledgeRepository(connection).add(
        Knowledge(
            knowledge_id=knowledge_id,
            memory_space_id=memory_space_id,
            title=title,
            target=target,
            statement=statement,
            rationale=rationale,
            phase="pattern",
            version=1,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            superseded_by_id=superseded_by_id,
            deleted_at=deleted_at,
        )
    )


def _add_atom(
    connection,
    *,
    memory_space_id: str,
    atom_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO atoms (
            atom_id, memory_space_id, source_system, source_instance_id,
            source_profile_id, source_session_id, source_round_id,
            source_round_key, source_digest, source_schema_version,
            resolver_version, extraction_host, extraction_provider,
            extraction_model, extraction_prompt_version,
            extraction_schema_version, extraction_purpose,
            title, target, statement, rationale, result, artifacts_json,
            content_digest, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            atom_id,
            memory_space_id,
            "hermes",
            "instance-1",
            "profile-1",
            "session-1",
            "round-1",
            f"{memory_space_id}:round-1",
            "sha256:" + "a" * 64,
            1,
            1,
            "hermes",
            "",
            "",
            1,
            1,
            "ledgermind.atom.extract",
            "Title",
            "target",
            "statement",
            "",
            "",
            "[]",
            "sha256:" + "b" * 64,
            "2026-01-01T00:00:00+00:00",
        ),
    )


def _add_evidence(
    connection,
    *,
    knowledge_id: str,
    atom_id: str,
    relation: str,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    SQLiteEvidenceRepository(connection).add(
        KnowledgeEvidence(
            knowledge_id=knowledge_id,
            atom_id=atom_id,
            relation=relation,
            created_at=created_at,
        )
    )


def _emit_projection_event(
    projection: KnowledgeMarkdownProjection,
    *,
    event_type: str,
    memory_space_id: str,
    payload: dict[str, object] | None = None,
    aggregate_id: str = "k-default",
) -> bool:
    payload_json = json.dumps(
        {
            "event_type": event_type,
            "aggregate_id": aggregate_id,
            **(payload or {}),
        }
    )
    return projection.handle_event(
        event_type=event_type,
        memory_space_id=memory_space_id,
        aggregate_id=aggregate_id,
        payload_json=payload_json,
    )


def _safe_name(value: str) -> str:
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")
    return encoded or "_"


def _projection_file(root: Path, memory_space_id: str, knowledge_id: str) -> Path:
    return root / "knowledge" / _safe_name(memory_space_id) / f"{_safe_name(knowledge_id)}.md"


def test_projection_generates_deterministic_markdown_with_multiline_body(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-markdown",
        title="Knowledge with multiline",
        target="Target",
        statement="Statement line one\nStatement line two",
        rationale="Rationale line one\nRationale line two",
    )

    root = tmp_path / "markdown"
    projection = KnowledgeMarkdownProjection(connection=connection, markdown_root=root)

    changed_first = _emit_projection_event(
        projection,
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-markdown",
    )
    changed_second = _emit_projection_event(
        projection,
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        payload={"knowledge_ids": ["k-markdown"]},
        aggregate_id="k-markdown",
    )

    path = _projection_file(root, "space-a", "k-markdown")
    content = path.read_text(encoding="utf-8")

    assert changed_first is True
    assert changed_second is True
    assert path.exists()
    assert "knowledge_id: k-markdown" in content
    assert "memory_space_id: space-a" in content
    assert "# Knowledge with multiline" in content
    assert "Statement line one" in content
    assert "Statement line two" in content
    assert "Rationale line one" in content
    assert "Rationale line two" in content


def test_projection_includes_source_atoms_in_frontmatter(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-source",
        title="Knowledge with atoms",
        target="target",
        statement="statement",
    )
    _add_atom(
        connection,
        memory_space_id="space-a",
        atom_id="a-origin",
    )
    _add_evidence(
        connection,
        knowledge_id="k-source",
        atom_id="a-origin",
        relation=EvidenceRelation.ORIGIN,
    )

    projection = KnowledgeMarkdownProjection(connection=connection, markdown_root=tmp_path / "markdown")
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-source",
        payload_json=json.dumps({"event_type": KnowledgeCreated.EVENT_NAME, "knowledge_id": "k-source"}),
    )

    content = _projection_file(tmp_path / "markdown", "space-a", "k-source").read_text(encoding="utf-8")
    assert "- a-origin" in content


def test_projection_handles_created_superseded_and_deleted_events(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-old",
        title="Old",
        target="target",
        statement="old statement",
        rationale="old rationale",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-new",
        title="New",
        target="target",
        statement="new statement",
        rationale="new rationale",
    )

    projection = KnowledgeMarkdownProjection(connection=connection, markdown_root=tmp_path / "markdown")

    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-old",
        payload_json=json.dumps(
            {
                "event_type": KnowledgeCreated.EVENT_NAME,
                "knowledge_id": "k-old",
            }
        ),
    )
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-new",
        payload_json=json.dumps(
            {
                "event_type": KnowledgeCreated.EVENT_NAME,
                "knowledge_id": "k-new",
            }
        ),
    )

    changed = projection.handle_event(
        event_type=KnowledgeSuperseded.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-old",
        payload_json=json.dumps(
            {
                "event_type": KnowledgeSuperseded.EVENT_NAME,
                "previous_knowledge_id": "k-old",
                "next_knowledge_id": "k-new",
            }
        ),
    )
    assert changed is True
    assert _projection_file(tmp_path / "markdown", "space-a", "k-old").exists() is False
    assert _projection_file(tmp_path / "markdown", "space-a", "k-new").exists() is True

    removed = projection.handle_event(
        event_type=KnowledgeDeleted.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-new",
        payload_json=json.dumps({"event_type": KnowledgeDeleted.EVENT_NAME, "knowledge_id": "k-new"}),
    )
    assert removed is True
    assert _projection_file(tmp_path / "markdown", "space-a", "k-new").exists() is False


def test_projection_rebuild_empty_database(tmp_path) -> None:
    db = tmp_path / "state.db"
    _bootstrap(db)
    projection = KnowledgeMarkdownProjection(
        connection=_build_connection(db),
        markdown_root=tmp_path / "markdown",
    )

    count = projection.rebuild()
    assert count == 0
    assert not list((tmp_path / "markdown").rglob("*.md"))


def test_projection_rebuild_uses_only_current_records(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-current",
        title="Keep",
        target="target",
        statement="current",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-superseded",
        title="Skip",
        target="target",
        statement="old",
        superseded_by_id="k-current",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-deleted",
        title="Skip",
        target="target",
        statement="removed",
        deleted_at="2026-01-01T00:00:00+00:00",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-other",
        title="Keep2",
        target="target",
        statement="current2",
    )

    projection = KnowledgeMarkdownProjection(
        connection=connection,
        markdown_root=tmp_path / "markdown",
    )

    rebuilt = projection.rebuild()
    assert rebuilt == 2

    files = list((tmp_path / "markdown").rglob("*.md"))
    assert len(files) == 2
    assert {path.name for path in files} == {_safe_name("k-current") + ".md", _safe_name("k-other") + ".md"}


def test_projection_rebuild_is_atomic_on_failure(tmp_path, monkeypatch) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "space-a")
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-a",
        title="First",
        target="target",
        statement="one",
    )
    _add_knowledge(
        connection,
        memory_space_id="space-a",
        knowledge_id="k-b",
        title="Second",
        target="target",
        statement="two",
    )

    markdown_root = tmp_path / "markdown"
    projection = KnowledgeMarkdownProjection(connection=connection, markdown_root=markdown_root)
    assert projection.rebuild() == 2

    baseline = {path: path.read_text(encoding="utf-8") for path in markdown_root.rglob("*.md")}
    original = projection._write_atomic
    calls = {"count": 0}

    def _failing(path: Path, content: str) -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("interrupted markdown rebuild")
        original(path, content)

    monkeypatch.setattr(projection, "_write_atomic", _failing)

    with pytest.raises(RuntimeError, match="interrupted markdown rebuild"):
        projection.rebuild()

    assert baseline == {path: path.read_text(encoding="utf-8") for path in markdown_root.rglob("*.md")}


def test_projection_file_path_is_safe_for_memory_space(tmp_path) -> None:
    db = tmp_path / "state.db"
    connection = _build_connection(db)
    _ensure_space(connection, "team/unsafe:space")
    _add_knowledge(
        connection,
        memory_space_id="team/unsafe:space",
        knowledge_id="k-safe",
        title="Safe",
        target="target",
        statement="ok",
    )

    projection = KnowledgeMarkdownProjection(connection=connection, markdown_root=tmp_path / "markdown")
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="team/unsafe:space",
        aggregate_id="k-safe",
        payload_json=json.dumps({"event_type": KnowledgeCreated.EVENT_NAME, "knowledge_id": "k-safe"}),
    )

    folder = list((tmp_path / "markdown" / "knowledge").iterdir())[0]
    assert folder.is_dir()
    assert "/" not in folder.name
    assert folder.name == _safe_name("team/unsafe:space")
