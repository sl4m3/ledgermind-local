"""Integration tests for Markdown git-audit projection."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from ledgermind_core.domain.events import KnowledgeCreated

from ledgermind_local.projections import KnowledgeMarkdownGitAuditProjection


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")


def _init_markdown_file(root: Path) -> None:
    target = root / "knowledge" / "c3Jj" / "k-1.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# Example\n", encoding="utf-8")


def _git_log_count(root: Path) -> int:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    return int((result.stdout or "0").strip() or 0)


def test_git_audit_projection_skips_when_disabled(tmp_path: Path) -> None:
    root = tmp_path / "markdown"
    _init_markdown_file(root)
    projection = KnowledgeMarkdownGitAuditProjection(markdown_root=root, enabled=False)

    changed = projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-1",
        payload_json='{"event_type":"knowledge.created","aggregate_id":"k-1"}',
    )

    assert changed is False
    assert not (root / ".git").exists()
    assert projection.rebuild() == 0


def test_git_audit_projection_batches_events(tmp_path: Path) -> None:
    _require_git()
    root = tmp_path / "markdown"
    _init_markdown_file(root)

    projection = KnowledgeMarkdownGitAuditProjection(
        markdown_root=root,
        enabled=True,
        batch_size=2,
    )

    assert projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-1",
        payload_json='{"event_type":"knowledge.created","aggregate_id":"k-1"}',
    ) is False
    assert _git_log_count(root) == 0

    (root / "knowledge" / "c3Jj" / "k-2.md").write_text("# Another\n", encoding="utf-8")
    assert projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-2",
        payload_json='{"event_type":"knowledge.created","aggregate_id":"k-2"}',
    ) is True
    assert _git_log_count(root) == 1


def test_git_audit_projection_nothing_to_commit_is_not_an_error(tmp_path: Path) -> None:
    _require_git()
    root = tmp_path / "markdown"
    root.mkdir(parents=True)

    projection = KnowledgeMarkdownGitAuditProjection(markdown_root=root, enabled=True, batch_size=1)

    assert projection.rebuild() == 0
    assert projection.last_error is None
    assert (root / ".git").exists()
    assert (root / ".gitignore").exists()


def test_git_audit_projection_ignores_sensitive_files(tmp_path: Path) -> None:
    _require_git()
    root = tmp_path / "markdown"
    _init_markdown_file(root)

    projection = KnowledgeMarkdownGitAuditProjection(markdown_root=root, enabled=True, batch_size=1)
    assert projection.rebuild() == 1

    gitignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "server.token" in gitignore
    assert "*.token" in gitignore
    assert ".token" in gitignore
    assert "*.db" in gitignore
    assert "*.db-shm" in gitignore
    assert "*.db-wal" in gitignore
    assert "*.log" in gitignore
    assert "*.logs" in gitignore

    (root / "secret.token").write_text("token", encoding="utf-8")
    (root / "state.db").write_text("{}", encoding="utf-8")
    (root / "history.log").write_text("log", encoding="utf-8")
    projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-1",
        payload_json='{"event_type":"knowledge.created","aggregate_id":"k-1"}',
    )

    for ignored in ("secret.token", "state.db", "history.log"):
        check = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--quiet", ignored],
            capture_output=True,
            text=True,
            check=False,
        )
        assert check.returncode == 0, f"{ignored} is not ignored by git"


def test_git_audit_projection_reports_missing_git_without_crash(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "markdown"
    _init_markdown_file(root)
    monkeypatch.setattr("ledgermind_local.projections.git_audit.shutil.which", lambda _name: None)

    projection = KnowledgeMarkdownGitAuditProjection(markdown_root=root, enabled=True, batch_size=1)
    assert projection.rebuild() == 0
    assert projection.last_error is not None
    assert "git is not available" in projection.last_error


def test_git_audit_projection_records_git_error_without_crash(tmp_path: Path, monkeypatch) -> None:
    _require_git()
    root = tmp_path / "markdown"
    _init_markdown_file(root)

    projection = KnowledgeMarkdownGitAuditProjection(markdown_root=root, enabled=True, batch_size=1)
    original_run = projection._run

    def _run(args: list[str], check: bool = True):  # type: ignore[override]
        if args and args[0] == "commit":
            return subprocess.CompletedProcess(args, 1, "", "simulated git failure")
        return original_run(args, check=check)

    monkeypatch.setattr(projection, "_run", _run)

    changed = projection.handle_event(
        event_type=KnowledgeCreated.EVENT_NAME,
        memory_space_id="space-a",
        aggregate_id="k-1",
        payload_json='{"event_type":"knowledge.created","aggregate_id":"k-1"}',
    )

    assert changed is False
    assert projection.last_error is not None
    assert "simulated git failure" in projection.last_error
