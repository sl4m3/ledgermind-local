"""Deterministic Markdown projection driven by Rust Core events."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_DELETE,
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    ProjectionDeletePayload,
    ProjectionUpsertPayload,
)

_PROJECTION_NAME = "projections.markdown"
_PROJECTION_VERSION = 1


class KnowledgeMarkdownProjection:
    """Render public Core projection payloads into Local Markdown files."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    def __init__(self, *, connection: object, markdown_root: str | Path) -> None:
        self._connection = connection
        self._markdown_root = Path(markdown_root)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else str(value)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _safe_name(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return "_"
        encoded = (
            base64.urlsafe_b64encode(normalized.encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        return encoded or "_"

    def _entry_path(self, memory_space_id: str, knowledge_id: str) -> Path:
        return (
            self._markdown_root
            / "knowledge"
            / self._safe_name(memory_space_id)
            / f"{self._safe_name(knowledge_id)}.md"
        )

    @staticmethod
    def _remove_empty_parents(path: Path) -> None:
        current = path.parent
        stop = path.parent.parent
        while current != stop:
            try:
                entries = list(current.iterdir())
            except FileNotFoundError:
                return
            if entries:
                return
            current.rmdir()
            current = current.parent

    def _write_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as temp:
                temp_name = temp.name
                temp.write(content)
            os.replace(temp_name, path)
        finally:
            if temp_name is not None and Path(temp_name).exists():
                Path(temp_name).unlink()

    def _remove(self, memory_space_id: str, knowledge_id: str) -> bool:
        path = self._entry_path(memory_space_id, knowledge_id)
        if not path.exists():
            return False
        path.unlink()
        self._remove_empty_parents(path)
        return True

    def _render_core_payload(self, payload: ProjectionUpsertPayload) -> str:
        metadata = yaml.safe_dump(
            {
                "knowledge_id": payload.knowledge_id,
                "memory_space_id": payload.memory_space_id,
                "projection_version": payload.projection_version,
                "statement": payload.statement,
                "target": payload.target,
                "title": payload.title,
            },
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        )
        return (
            "---\n"
            f"{metadata}"
            "---\n\n"
            f"# {self._normalize_text(payload.title)}\n\n"
            f"## Утверждение\n{self._normalize_text(payload.statement)}\n"
        )

    def handle_core_event(self, event: CoreProjectionEvent) -> bool:
        """Apply a Core event without reading any canonical database row."""

        parsed = event.parse_payload()
        if event.event_type == CORE_PROJECTION_UPSERT:
            if not isinstance(parsed, ProjectionUpsertPayload):
                raise TypeError("upsert event did not produce an upsert payload")
            self._write_atomic(
                self._entry_path(parsed.memory_space_id, parsed.knowledge_id),
                self._render_core_payload(parsed),
            )
            return True
        if event.event_type == CORE_PROJECTION_DELETE:
            if not isinstance(parsed, ProjectionDeletePayload):
                raise TypeError("delete event did not produce a delete payload")
            return self._remove(parsed.memory_space_id, parsed.knowledge_id)
        raise ValueError(f"unsupported Core projection event type: {event.event_type}")


__all__ = ["KnowledgeMarkdownProjection"]