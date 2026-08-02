"""Knowledge vector projection for local vector search."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from ledgermind_core.domain.events import (
    KnowledgeCreated,
    KnowledgeDeleted,
    KnowledgeSuperseded,
)
from typing_extensions import Self

from ledgermind_local.persistence import SQLiteKnowledgeRepository
from ledgermind_local.projections.vector_store import VectorProjectionStore
from ledgermind_local.projections.vectorizer import Vectorizer

_PROJECTION_NAME = "projections.knowledge"
_PROJECTION_VERSION = 1


class KnowledgeVectorProjection:
    """Project local knowledge updates into a vector index."""

    projection_name = _PROJECTION_NAME
    projection_version = _PROJECTION_VERSION

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        vector_store_root: str | Path,
        vectorizer_factory: Callable[[], Vectorizer],
    ) -> None:
        self._connection = connection
        self._repository = SQLiteKnowledgeRepository(connection=connection)
        self._vector_store_root = Path(vector_store_root)
        self._vector_store = VectorProjectionStore(self._vector_store_root)
        self._vectorizer_factory = vectorizer_factory
        self._vectorizer_instance: Vectorizer | None = None

    @property
    def _vectorizer(self) -> Vectorizer:
        if self._vectorizer_instance is None:
            vectorizer = self._vectorizer_factory()
            self._vectorizer_instance = vectorizer
        return self._vectorizer_instance

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._vectorizer_instance is not None:
            self._vectorizer_instance.close()
            self._vectorizer_instance = None

    def rebuild(self, *, memory_space_id: str | None = None, batch_size: int = 64) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        vectorizer = self._vectorizer
        records = self._current_records(memory_space_id=memory_space_id)

        knowledge_ids: list[str] = []
        vectors: list[Sequence[float]] = []

        candidate_root = Path(f"{self._vector_store_root}.rebuild")
        if candidate_root.exists():
            if candidate_root.is_file():
                candidate_root.unlink()
            else:
                shutil.rmtree(candidate_root)

        try:
            for offset in range(0, len(records), batch_size):
                batch = records[offset : offset + batch_size]
                texts = [text for _knowledge_id, text in batch]
                raw_vectors = list(vectorizer.encode(texts))
                if len(raw_vectors) != len(batch):
                    raise ValueError("partial vectorization result")
                knowledge_ids.extend(knowledge_id for knowledge_id, _ in batch)
                vectors.extend(self._coerce_vector(raw) for raw in raw_vectors)

            candidate_store = VectorProjectionStore(
                candidate_root,
                model_fingerprint=vectorizer.fingerprint,
                model_name=getattr(vectorizer, "model_name", None),
                model_dimension=vectorizer.dimension,
            )
            candidate_store.rebuild(knowledge_ids, vectors)
            self._replace_vector_store(candidate_root, self._vector_store_root)

            self._vector_store = VectorProjectionStore(
                self._vector_store_root,
                model_fingerprint=vectorizer.fingerprint,
                model_name=getattr(vectorizer, "model_name", None),
                model_dimension=vectorizer.dimension,
            )
            return len(knowledge_ids)
        except Exception:
            if candidate_root.exists():
                if candidate_root.is_file():
                    candidate_root.unlink()
                else:
                    shutil.rmtree(candidate_root, ignore_errors=True)
            raise

    def _replace_vector_store(self, source_root: Path, target_root: Path) -> None:
        if not source_root.exists():
            raise RuntimeError("candidate vector store was not created")

        if not target_root.exists():
            os.replace(source_root, target_root)
            return

        backup = Path(f"{target_root}.{os.getpid()}.old")
        if backup.exists():
            if backup.is_file():
                backup.unlink()
            else:
                shutil.rmtree(backup)

        os.replace(target_root, backup)
        try:
            os.replace(source_root, target_root)
        except Exception:
            os.replace(backup, target_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def _current_records(
        self,
        *,
        memory_space_id: str | None = None,
    ) -> list[tuple[str, str]]:
        query = """
            SELECT
                knowledge_id,
                memory_space_id,
                title,
                target,
                statement,
                rationale
            FROM knowledge_items
            WHERE superseded_by_id IS NULL
              AND deleted_at IS NULL
        """
        if memory_space_id is None:
            query = f"{query} ORDER BY memory_space_id ASC, knowledge_id ASC"
            rows = self._connection.execute(query).fetchall()
        else:
            query = f"{query} AND memory_space_id = ? ORDER BY knowledge_id ASC"
            rows = self._connection.execute(query, (memory_space_id,)).fetchall()

        return [(row["knowledge_id"], self._build_text(row)) for row in rows]

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

        payload = self._extract_payload(payload_json)
        normalized_event = self._coerce_string(payload.get("event_type")) or event_type

        if normalized_event in (
            KnowledgeCreated.EVENT_NAME,
            "knowledge.updated",
        ):
            knowledge_ids = self._coerce_ids(
                payload.get("knowledge_ids"),
            )
            if not knowledge_ids:
                candidate = (
                    self._coerce_string(payload.get("knowledge_id"))
                    or self._coerce_string(payload.get("aggregate_id"))
                    or self._coerce_string(aggregate_id)
                )
                knowledge_ids = [candidate] if candidate is not None else []
            return self._apply_changes(remove_ids=(), upsert_ids=knowledge_ids, memory_space_id=memory_space_id)

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
            return self._apply_changes(
                remove_ids=[knowledge_id],
                upsert_ids=(),
                memory_space_id=memory_space_id,
            )

        return False

    def _apply_changes(
        self,
        *,
        remove_ids: Sequence[str | None],
        upsert_ids: Sequence[str | None],
        memory_space_id: str,
    ) -> bool:
        remove = self._unique_ids(remove_ids)
        upsert = self._unique_ids(upsert_ids)

        if remove:
            self._vector_store.remove([knowledge_id for knowledge_id in remove if knowledge_id not in set(upsert)])

        upsert_payload = self._load_current_knowledge(memory_space_id, upsert)
        if upsert_payload:
            vectorizer = self._vectorizer
            knowledge_ids = [knowledge_id for knowledge_id, _ in upsert_payload]
            texts = [self._build_text(item) for _, item in upsert_payload]
            vectors = vectorizer.encode(texts)
            if len(vectors) != len(knowledge_ids):
                raise ValueError("partial vectorization result")
            self._vector_store.upsert(knowledge_ids, vectors)

        if self._vector_store.dirty:
            self._vector_store.flush()
            return True

        return False

    def _load_current_knowledge(self, memory_space_id: str, knowledge_ids: Sequence[str]) -> list[tuple[str, Any]]:
        if not knowledge_ids:
            return []

        rows: list[tuple[str, Any]] = []
        for knowledge_id in knowledge_ids:
            knowledge = self._repository.get(knowledge_id=knowledge_id, memory_space_id=memory_space_id)
            if knowledge is None:
                continue
            if knowledge.superseded_by_id is not None:
                continue
            if knowledge.deleted_at is not None:
                continue
            rows.append((knowledge_id, knowledge))
        return rows

    @staticmethod
    def _coerce_vector(value: object) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("embedding must be a sequence of numbers")  # noqa: TRY004 - vector validation contract
        return [float(item) for item in value]

    @staticmethod
    def _build_text(knowledge: object) -> str:
        title = _value(knowledge, "title")
        target = _value(knowledge, "target")
        statement = _value(knowledge, "statement")
        rationale = _value(knowledge, "rationale")
        return " ".join(
            value
            for value in (title, target, statement, rationale)
            if isinstance(value, str) and value
        )

    @staticmethod
    def _coerce_ids(values: object) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return [item for item in (KnowledgeVectorProjection._coerce_string(item) for item in values) if item is not None]
        coerce = KnowledgeVectorProjection._coerce_string(values)
        return [coerce] if coerce is not None else []

    @staticmethod
    def _coerce_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized if normalized else None

    @staticmethod
    def _unique_ids(values: Sequence[str | None]) -> tuple[str, ...]:
        ordered = [value for value in values if value is not None]
        return tuple(dict.fromkeys(ordered).keys())

    @staticmethod
    def _extract_payload(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


def _value(obj: object, key: str) -> object:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "__getitem__"):
        try:
            return obj[key]
        except (TypeError, KeyError):
            pass
    return getattr(obj, key, None)


__all__ = [
    "KnowledgeVectorProjection",
]
