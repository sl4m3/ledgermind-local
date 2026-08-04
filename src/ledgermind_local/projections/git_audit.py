"""Git audit projection for Markdown knowledge files."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

from ledgermind_local.core_gateway.event_contracts import (
    KnowledgeCreated,
    KnowledgeDeleted,
    KnowledgeSuperseded,
)

_PROJECTION_NAME = "projections.markdown_audit"
_PROJECTION_VERSION = 1
_KNOWN_EVENTS: tuple[str, ...] = (
    KnowledgeCreated.EVENT_NAME,
    KnowledgeSuperseded.EVENT_NAME,
    KnowledgeDeleted.EVENT_NAME,
    "knowledge.updated",
)


class KnowledgeMarkdownGitAuditProjection:
    """Store Markdown projection history in a local Git repository."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    _IGNORE_LINES = (
        "server.token",
        "*.db",
        "*.db-journal",
        "*.db-shm",
        "*.db-wal",
        "*.log",
        "*.logs",
        ".token",
        "*.token",
        ".env",
        "logs/",
    )

    def __init__(
        self,
        *,
        markdown_root: str | Path,
        enabled: bool = False,
        batch_size: int = 16,
        git_author_name: str = "LedgerMind Local",
        git_author_email: str = "local-audit@ledgermind.internal",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._enabled = bool(enabled)
        self._markdown_root = Path(markdown_root)
        self._batch_size = batch_size
        self._git_author_name = git_author_name.strip() or "LedgerMind Local"
        self._git_author_email = (
            git_author_email.strip() or "local-audit@ledgermind.internal"
        )
        self._pending = 0
        self._last_error: str | None = None
        self._git_binary = shutil.which("git")
        self._repo_ready = False

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str | None = None,
    ) -> bool:
        del memory_space_id
        del aggregate_id

        if not self._enabled:
            return False

        payload = self._extract_payload(payload_json)
        normalized_event = self._coerce_string(payload.get("event_type")) or event_type
        if normalized_event not in _KNOWN_EVENTS:
            return False

        self._pending += 1
        if self._pending < self._batch_size:
            return False

        return cast(bool, self._flush_batch(need_changes=True))

    def rebuild(self, *, memory_space_id: str | None = None) -> int:
        del memory_space_id

        if not self._enabled:
            return 0

        return self._flush_batch(need_changes=False)

    def close(self) -> None:
        if self._enabled and self._pending:
            self._flush_batch(need_changes=False)

    def _flush_batch(self, need_changes: bool) -> bool | int:
        if not self._enabled:
            return False
        try:
            self._last_error = None
            self._ensure_repository()
            staged = self._stage_all()
            audit_staged = self._auditable_changes(staged)
            if not audit_staged and need_changes:
                self._pending = 0
                return False

            if not audit_staged:
                return 0

            self._commit(staged)
            self._pending = 0
            self._last_error = None
            if need_changes:
                return True
            return len(audit_staged)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._pending = 0
            return False if need_changes else 0

    def _ensure_repository(self) -> None:
        if self._git_binary is None:
            raise RuntimeError("git is not available")

        self._markdown_root.mkdir(parents=True, exist_ok=True)
        if self._repo_ready:
            return

        if not self._has_git_repo():
            self._run(["init"])
        self._set_local_author()
        self._write_gitignore()
        self._repo_ready = True

    def _has_git_repo(self) -> bool:
        proc = self._run(["rev-parse", "--is-inside-work-tree"], check=False)
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def _write_gitignore(self) -> None:
        gitignore = self._markdown_root / ".gitignore"
        current = self._read_lines(gitignore)
        lines = list(current)
        for item in self._IGNORE_LINES:
            if item not in lines:
                lines.append(item)
        desired = "\n".join(lines).strip()
        if desired:
            desired = f"{desired}\n"

        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if existing != desired:
            gitignore.write_text(desired, encoding="utf-8")
        self._run(["add", ".gitignore"], check=False)

    def _stage_all(self) -> list[str]:
        self._run(["add", "-A"])
        staged = self._run(["diff", "--cached", "--name-only"])
        return [line.strip() for line in staged.stdout.splitlines() if line.strip()]

    @staticmethod
    def _auditable_changes(staged_files: list[str]) -> list[str]:
        return [path for path in staged_files if path != ".gitignore"]

    def _commit(self, staged_files: list[str]) -> None:
        if not staged_files:
            return
        message = self._build_commit_message(staged_files)
        commit = self._run(["commit", "--no-gpg-sign", "-m", message], check=False)
        if commit.returncode != 0:
            output = commit.stderr.strip() or commit.stdout.strip()
            if "nothing to commit" in output.lower():
                return
            if output:
                raise RuntimeError(output)
            raise RuntimeError("nothing to commit")

    def _set_local_author(self) -> None:
        self._run(
            ["config", "--local", "user.name", self._git_author_name], check=False
        )
        self._run(
            ["config", "--local", "user.email", self._git_author_email], check=False
        )

    def _build_commit_message(self, staged_files: list[str]) -> str:
        del staged_files
        return "chore(markdown): audit projection update"

    @staticmethod
    def _coerce_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized if normalized else None

    @staticmethod
    def _extract_payload(raw: str | None) -> dict[str, object]:
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [
            line.rstrip("\n")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _run(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        if self._git_binary is None:
            raise RuntimeError("git is not available")

        command = [self._git_binary, *args]
        result = subprocess.run(
            command,
            cwd=self._markdown_root,
            check=False,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        if check and result.returncode != 0:
            output = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(output or "git command failed")
        return result


MarkdownGitAuditProjection = KnowledgeMarkdownGitAuditProjection

__all__ = [
    "KnowledgeMarkdownGitAuditProjection",
    "MarkdownGitAuditProjection",
]
