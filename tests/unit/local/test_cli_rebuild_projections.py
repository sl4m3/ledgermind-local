"""Tests for `ledgermind rebuild-projections` command."""

from __future__ import annotations

from pathlib import Path

import bootstrap
import cli as cli_module
from config import LocalConfig
from persistence import (
    Knowledge,
    SQLiteKnowledgeRepository,
    migrations,
    open_sqlite_connection,
)
from projections import VectorProjectionStore


def _seed_kb_with_known_state(database_path: Path) -> None:
    connection = open_sqlite_connection(database_path)
    try:
        migrations.apply_migrations(connection)
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
                "space-a",
                None,
                "hermes",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        repository = SQLiteKnowledgeRepository(connection=connection)
        repository.add(
            Knowledge(
                knowledge_id="k-a",
                memory_space_id="space-a",
                title="Title",
                target="Target",
                statement="Statement",
                rationale="",
                phase="pattern",
                version=1,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                superseded_by_id=None,
                deleted_at=None,
            )
        )
        connection.commit()
    finally:
        connection.close()


class _SpyVectorizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.closed = 0

    @property
    def fingerprint(self) -> str:
        return "cli-vector"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: list[str]) -> list[tuple[float, float]]:
        batch = tuple(texts)
        self.calls.append(batch)
        return [(float(len(text)), float(len(texts))) for text in texts]

    def close(self) -> None:
        self.closed += 1


def test_command_rebuild_projections_runs_vector_rebuild(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(home=home)
    database_path = home / "ledgermind.db"
    _seed_kb_with_known_state(database_path)

    vectorizer = _SpyVectorizer()
    monkeypatch.setattr(cli_module, "_build_vectorizer_factory", lambda: lambda: vectorizer)

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["vector"]})()
    )

    assert result == 0
    assert vectorizer.calls

    store = VectorProjectionStore(cli_module._build_vector_store_root(database_path), model_dimension=2)
    assert store.ids == ("k-a",)


def test_command_rebuild_projections_runs_markdown_rebuild(tmp_path: Path) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(home=home)
    database_path = home / "ledgermind.db"
    _seed_kb_with_known_state(database_path)

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["markdown"]})()
    )

    assert result == 0
    root = cli_module._build_markdown_root(database_path) / "knowledge"
    files = list(root.rglob("*.md"))
    assert len(files) == 1
    payload = files[0].read_text(encoding="utf-8")
    assert "knowledge_id: k-a" in payload
    assert "# Title" in payload


def test_command_rebuild_projections_runs_fts_rebuild(tmp_path: Path) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(home=home)
    database_path = home / "ledgermind.db"
    _seed_kb_with_known_state(database_path)

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["fts"]})()
    )

    assert result == 0
    connection = open_sqlite_connection(database_path)
    try:
        row = connection.execute("SELECT COUNT(*) AS total FROM knowledge_fts").fetchone()
    finally:
        connection.close()
    assert row["total"] == 1


def test_command_rebuild_projections_runs_markdown_audit_when_enabled(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(
        home=home,
        config=LocalConfig(config_version=1, markdown_audit_enabled=True),
    )

    calls: dict[str, object] = {
        "init": 0,
        "rebuild": 0,
        "close": 0,
        "markdown_root": None,
        "enabled": None,
    }

    class _FakeMarkdownAuditProjection:
        def __init__(self, *args, **kwargs) -> None:
            calls["init"] = 1
            calls["markdown_root"] = kwargs["markdown_root"]
            calls["enabled"] = kwargs.get("enabled", False)

        def rebuild(self) -> int:
            calls["rebuild"] = 1
            return 2

        def close(self) -> None:
            calls["close"] = 1

    monkeypatch.setattr(cli_module, "KnowledgeMarkdownGitAuditProjection", _FakeMarkdownAuditProjection)

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["markdown_audit"]})()
    )

    assert result == 0
    assert calls["init"] == 1
    assert calls["rebuild"] == 1
    assert calls["close"] == 1
    assert calls["markdown_root"] is not None
    assert calls["enabled"] is True


def test_command_rebuild_projections_rejects_unknown_projection(tmp_path: Path) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(home=home)

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["unknown"]})()
    )

    assert result == 2


def test_command_rebuild_projections_rejects_markdown_audit_when_disabled(
    tmp_path: Path,
) -> None:
    home = tmp_path / "service"
    bootstrap.initialize_local_layout(
        home=home,
        config=LocalConfig(config_version=1, markdown_audit_enabled=False),
    )

    result = cli_module._command_rebuild_projections(
        type("Args", (), {"home": str(home), "only": ["markdown_audit"]})()
    )

    assert result == 2
