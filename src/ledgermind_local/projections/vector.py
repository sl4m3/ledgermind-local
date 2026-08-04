"""Knowledge vector projection driven by Rust Core events."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from typing_extensions import Self

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_DELETE,
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    ProjectionDeletePayload,
    ProjectionUpsertPayload,
)
from ledgermind_local.projections.vector_store import VectorProjectionStore
from ledgermind_local.projections.vectorizer import Vectorizer

_PROJECTION_NAME = "projections.knowledge"
_PROJECTION_VERSION = 1


class KnowledgeVectorProjection:
    """Project public Core payloads into a Local vector index."""

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
        self._vector_store = VectorProjectionStore(Path(vector_store_root))
        self._vectorizer_factory = vectorizer_factory
        self._vectorizer_instance: Vectorizer | None = None

    @property
    def _vectorizer(self) -> Vectorizer:
        if self._vectorizer_instance is None:
            self._vectorizer_instance = self._vectorizer_factory()
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

    def handle_core_event(self, event: CoreProjectionEvent) -> bool:
        """Apply a Core event without reading any canonical database row."""

        parsed = event.parse_payload()
        if event.event_type == CORE_PROJECTION_UPSERT:
            if not isinstance(parsed, ProjectionUpsertPayload):
                raise TypeError("upsert event did not produce an upsert payload")
            vectorizer = self._vectorizer
            text = self._build_text(
                {
                    "title": parsed.title,
                    "target": parsed.target,
                    "statement": parsed.statement,
                }
            )
            vectors = list(vectorizer.encode([text]))
            if len(vectors) != 1:
                raise ValueError("partial vectorization result")
            self._vector_store.upsert(
                [parsed.knowledge_id], [self._coerce_vector(vectors[0])]
            )
            self._vector_store.flush()
            return True
        if event.event_type == CORE_PROJECTION_DELETE:
            if not isinstance(parsed, ProjectionDeletePayload):
                raise TypeError("delete event did not produce a delete payload")
            self._vector_store.remove([parsed.knowledge_id])
            self._vector_store.flush()
            return True
        raise ValueError(f"unsupported Core projection event type: {event.event_type}")

    @staticmethod
    def _coerce_vector(value: object) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise TypeError("embedding must be a sequence of numbers")
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


def _value(obj: object, key: str) -> object:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "__getitem__"):
        try:
            return obj[key]  # type: ignore[index]
        except (TypeError, KeyError):
            pass
    return getattr(obj, key, None)


__all__ = ["KnowledgeVectorProjection"]