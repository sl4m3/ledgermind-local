"""Deterministic Markdown projection for knowledge items."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from ledgermind_core.domain.events import (
    KnowledgeCreated,
    KnowledgeDeleted,
    KnowledgeSuperseded,
)

from ledgermind_local.persistence import (
    SQLiteEvidenceRepository,
    SQLiteKnowledgeRepository,
)

_PROJECTION_NAME = "projections.markdown"
_PROJECTION_VERSION = 1


class KnowledgeMarkdownProjection:
    """Projector that renders canonical knowledge rows into Markdown files."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    def __init__(self, *, connection: object, markdown_root: str | Path) -> None:
        self._connection = connection
        self._repository = SQLiteKnowledgeRepository(connection=connection)  # type: ignore[arg-type]
        self._evidence_repository = SQLiteEvidenceRepository(connection=connection)  # type: ignore[arg-type]
        self._markdown_root = Path(markdown_root)

    def _current_records(self, *, memory_space_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale,
                phase,
                version,
                created_at,
                updated_at,
                superseded_by_id
            FROM knowledge_items
            WHERE superseded_by_id IS NULL
              AND deleted_at IS NULL
        """
        if memory_space_id is None:
            query = f"{query} ORDER BY memory_space_id ASC, knowledge_id ASC"
            rows = self._connection.execute(query).fetchall()  # type: ignore[attr-defined]
        else:
            query = f"{query} AND memory_space_id = ? ORDER BY knowledge_id ASC"
            rows = self._connection.execute(query, (memory_space_id,)).fetchall()  # type: ignore[attr-defined]

        return [dict(row) for row in rows]

    def _render_frontmatter(self, knowledge: dict[str, Any]) -> str:
        metadata: dict[str, Any] = {
            "knowledge_id": knowledge["knowledge_id"],
            "memory_space_id": knowledge["memory_space_id"],
            "title": knowledge["title"],
            "target": knowledge["target"],
            "statement": knowledge["statement"],
            "rationale": knowledge["rationale"],
            "phase": knowledge["phase"],
            "version": int(knowledge["version"]),
            "created_at": knowledge["created_at"],
            "updated_at": knowledge["updated_at"],
            "superseded_by_id": knowledge["superseded_by_id"],
            "source_atoms": knowledge.get("source_atoms", []),
        }
        serialized = cast(str | None, yaml.safe_dump(
            metadata,
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        ))
        if serialized is None:
            serialized = ""
        if serialized.endswith("\n"):
            return serialized
        return f"{serialized}\n"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else str(value)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _render_markdown(self, knowledge: dict[str, Any]) -> str:
        title = self._normalize_text(knowledge["title"])
        statement = self._normalize_text(knowledge["statement"])
        rationale = self._normalize_text(knowledge["rationale"])
        heading = title if title else "(untitled)"

        return (
            "---\n"
            f"{self._render_frontmatter(knowledge)}"
            "---\n\n"
            f"# {heading}\n\n"
            f"## Утверждение\n{statement}\n\n"
            "## Обоснование\n"
            f"{rationale}\n"
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return "_"
        encoded = base64.urlsafe_b64encode(normalized.encode("utf-8")).decode("ascii").rstrip("=")
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
        path = self._entry_path(memory_space_id=memory_space_id, knowledge_id=knowledge_id)
        if not path.exists():
            return False
        path.unlink()
        self._remove_empty_parents(path)
        return True

    def _upsert(self, knowledge: dict[str, Any]) -> bool:
        path = self._entry_path(
            memory_space_id=knowledge["memory_space_id"],
            knowledge_id=knowledge["knowledge_id"],
        )
        self._write_atomic(path, self._render_markdown(knowledge))
        return True

    @staticmethod
    def _coerce_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized if normalized else None

    @staticmethod
    def _coerce_ids(values: object) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (tuple, list)):
            return [
                item
                for item in (
                    KnowledgeMarkdownProjection._coerce_string(item) for item in values
                )
                if item is not None
            ]
        normalized = KnowledgeMarkdownProjection._coerce_string(values)
        return [normalized] if normalized is not None else []

    @staticmethod
    def _coerce_payload(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_current_knowledge(self, memory_space_id: str, knowledge_id: str) -> dict[str, Any] | None:
        knowledge = self._repository.get(knowledge_id=knowledge_id, memory_space_id=memory_space_id)
        if knowledge is None:
            return None
        if knowledge.superseded_by_id is not None or knowledge.deleted_at is not None:
            return None
        source_atoms = self._evidence_repository.list_atom_ids(
            memory_space_id=memory_space_id,
            knowledge_id=knowledge_id,
        )
        return {
            "knowledge_id": knowledge.knowledge_id,
            "memory_space_id": knowledge.memory_space_id,
            "title": knowledge.title,
            "target": knowledge.target,
            "statement": knowledge.statement,
            "rationale": knowledge.rationale,
            "phase": knowledge.phase,
            "version": knowledge.version,
            "created_at": knowledge.created_at,
            "updated_at": knowledge.updated_at,
            "superseded_by_id": knowledge.superseded_by_id,
            "source_atoms": source_atoms,
        }

    def _apply_changes(
        self,
        *,
        remove_ids: Sequence[str | None],
        upsert_ids: Sequence[str | None],
        memory_space_id: str,
    ) -> bool:
        changed = False

        remove = tuple(dict.fromkeys(item for item in remove_ids if item is not None))
        upsert = tuple(dict.fromkeys(item for item in upsert_ids if item is not None))

        for remove_id in remove:
            changed = self._remove(memory_space_id=memory_space_id, knowledge_id=remove_id) or changed

        for knowledge_id in upsert:
            knowledge = self._load_current_knowledge(memory_space_id=memory_space_id, knowledge_id=knowledge_id)
            if knowledge is None:
                changed = self._remove(memory_space_id=memory_space_id, knowledge_id=knowledge_id) or changed
                continue
            self._upsert(knowledge)
            changed = True

        return changed

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str | None = None,
    ) -> bool:
        if not memory_space_id:
            return False

        payload = self._coerce_payload(payload_json)
        normalized_event = self._coerce_string(payload.get("event_type")) or event_type

        if normalized_event in (
            KnowledgeCreated.EVENT_NAME,
            "knowledge.updated",
        ):
            knowledge_ids = self._coerce_ids(payload.get("knowledge_ids"))
            if not knowledge_ids:
                candidate = (
                    self._coerce_string(payload.get("knowledge_id"))
                    or self._coerce_string(payload.get("aggregate_id"))
                    or self._coerce_string(aggregate_id)
                )
                knowledge_ids = [candidate] if candidate is not None else []
            return self._apply_changes(
                remove_ids=(),
                upsert_ids=knowledge_ids,
                memory_space_id=memory_space_id,
            )

        if normalized_event == KnowledgeSuperseded.EVENT_NAME:
            remove_ids = self._coerce_ids(payload.get("previous_knowledge_id"))
            upsert_ids = self._coerce_ids(payload.get("next_knowledge_id"))
            return self._apply_changes(
                remove_ids=remove_ids,
                upsert_ids=upsert_ids,
                memory_space_id=memory_space_id,
            )

        if normalized_event == KnowledgeDeleted.EVENT_NAME:
            knowledge_id = (
                self._coerce_string(payload.get("knowledge_id"))
                or self._coerce_string(payload.get("aggregate_id"))
                or self._coerce_string(aggregate_id)
            )
            if knowledge_id is None:
                return False
            return self._remove(memory_space_id=memory_space_id, knowledge_id=knowledge_id)

        return False

    def _replace_markdown_root(self, candidate_root: Path, target_root: Path) -> None:
        if not candidate_root.exists():
            raise RuntimeError("candidate markdown root was not created")

        if not target_root.exists():
            os.replace(candidate_root, target_root)
            return

        backup = Path(f"{target_root}.{os.getpid()}.old")
        if backup.exists():
            if backup.is_file():
                backup.unlink()
            else:
                shutil.rmtree(backup)

        os.replace(target_root, backup)
        try:
            os.replace(candidate_root, target_root)
        except Exception:
            os.replace(backup, target_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def rebuild(self, *, memory_space_id: str | None = None) -> int:
        records = self._current_records(memory_space_id=memory_space_id)

        candidate_root = Path(f"{self._markdown_root}.rebuild")
        if candidate_root.exists():
            if candidate_root.is_file():
                candidate_root.unlink()
            else:
                shutil.rmtree(candidate_root)

        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
            for record in records:
                path = Path(f"{candidate_root}") / "knowledge" / self._safe_name(record["memory_space_id"]) / (
                    f"{self._safe_name(record['knowledge_id'])}.md"
                )
                self._write_atomic(path, self._render_markdown(record))

            self._replace_markdown_root(candidate_root, self._markdown_root)
            return len(records)
        except Exception:
            if candidate_root.exists():
                if candidate_root.is_file():
                    candidate_root.unlink()
                else:
                    shutil.rmtree(candidate_root, ignore_errors=True)
            raise


__all__ = ["KnowledgeMarkdownProjection"]
