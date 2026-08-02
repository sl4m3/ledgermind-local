"""Bootstrap utilities for the v4 local service."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from ledgermind_core.application.digests import calculate_source_round_key
from ledgermind_core.application.get_atom import GetAtomHandler
from ledgermind_core.application.get_knowledge import GetKnowledgeHandler
from ledgermind_core.application.ingest_atom import (
    IngestAtomHandler,
    JsonIngestAtomResultSerializer,
)
from ledgermind_core.application.retrieve_context import RetrieveContextHandler
from ledgermind_core.domain import (
    Atom,
    AtomContent,
    AtomId,
    ExtractionInfo,
    KnowledgeEvidence,
    KnowledgeId,
    KnowledgeItem,
    KnowledgeRevision,
    RevisionId,
    SourceReference,
)
from ledgermind_core.domain.policies import IsolatedPatternPolicy
from ledgermind_core.ports import (
    Clock,
    IdentifierFactory,
    KnowledgeSearch,
    UnitOfWork,
)
from ledgermind_core.ports.repository_ports import (
    AtomRepository,
    DomainEvent,
    EventRepository,
    EvidenceRepository,
    IdempotencyRepository,
    KnowledgeRepository,
    RevisionRepository,
    StoredIdempotencyResult,
)

from ledgermind_local.config import LocalConfig
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import (
    Atom as SqliteAtom,
)
from ledgermind_local.persistence import (
    Knowledge as SqliteKnowledge,
)
from ledgermind_local.persistence import (
    KnowledgeEvidence as SqliteKnowledgeEvidence,
)
from ledgermind_local.persistence import (
    KnowledgeRevision as SqliteKnowledgeRevision,
)
from ledgermind_local.persistence import (
    OutboxEvent as SqliteOutboxEvent,
)
from ledgermind_local.persistence import (
    SQLiteAtomRepository,
    SQLiteEvidenceRepository,
    SQLiteIdempotencyRepository,
    SQLiteKnowledgeRepository,
    SQLiteMemorySpaceRepository,
    SQLiteOutboxRepository,
    SQLiteRevisionRepository,
    SQLiteUnitOfWork,
    open_sqlite_connection,
)
from ledgermind_local.persistence import (
    StoredIdempotencyResult as SqliteStoredIdempotencyResult,
)
from ledgermind_local.search import (
    HybridKnowledgeSearchAdapter,
    SQLiteKnowledgeSearchAdapter,
)

DEFAULT_PROJECTIONS: tuple[str, ...] = (
    "projections.search",
    "projections.knowledge",
    "projections.markdown",
)


def build_projection_names(config: LocalConfig) -> tuple[str, ...]:
    names = list(DEFAULT_PROJECTIONS)
    if config.markdown_projection.enabled or config.markdown_audit_enabled:
        names.append("projections.markdown_audit")
    return tuple(names)


def _build_uow_factory(
    database_path: str | Path,
    *,
    busy_timeout_ms: int = 5_000,
    projection_names: tuple[str, ...] = DEFAULT_PROJECTIONS,
    write_transaction: bool = True,
) -> Callable[[], UnitOfWork]:
    clock = _SystemClock()
    identifiers = _UuidIdentifierFactory()

    def _factory() -> UnitOfWork:
        return _SQLiteCoreUnitOfWork(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=write_transaction,
            projection_names=projection_names,
            clock=clock,
            identifiers=identifiers,
        )

    return _factory

def _build_search_adapter(
    connection: sqlite3.Connection,
    database_path: str | Path,
) -> KnowledgeSearch:
    vector_store_root = _build_vector_store_root(database_path)
    factory = _build_vectorizer_factory()
    if not vector_store_root.exists() and factory is None:
        return SQLiteKnowledgeSearchAdapter(connection)
    return HybridKnowledgeSearchAdapter(
        connection=connection,
        vector_store_root=vector_store_root,
        vectorizer_factory=factory,
    )


def _build_vector_store_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".vectors")


def _build_markdown_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".markdown")


def _build_vectorizer_factory() -> Callable[[], Any] | None:
    model_path = os.environ.get("LEDGERMIND_VECTOR_MODEL_PATH")
    if not model_path:
        return None

    from ledgermind_local.projections import GGUFVectorizer

    return lambda: GGUFVectorizer(model_path=model_path)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.isoformat()


def _parse_timestamp(raw: str) -> datetime:
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        raise ValueError("stored timestamp must be timezone-aware")
    return timestamp


@dataclass(frozen=True, slots=True)
class GetKnowledgeHistoryQuery:
    memory_space_id: str
    knowledge_id: str

    def __post_init__(self) -> None:
        if not self.memory_space_id:
            raise ValueError("memory_space_id must not be empty")
        if not self.knowledge_id:
            raise ValueError("knowledge_id must not be empty")


@dataclass(frozen=True, slots=True)
class KnowledgeHistoryItem:
    revision_id: str
    version: int
    event_type: str
    snapshot: dict[str, object]
    cause_atom_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class GetKnowledgeHistoryResult:
    memory_space_id: str
    knowledge_id: str
    revisions: list[KnowledgeHistoryItem]


@dataclass(frozen=True, slots=True)
class GetKnowledgeEvidenceQuery:
    memory_space_id: str
    knowledge_id: str

    def __post_init__(self) -> None:
        if not self.memory_space_id:
            raise ValueError("memory_space_id must not be empty")
        if not self.knowledge_id:
            raise ValueError("knowledge_id must not be empty")


@dataclass(frozen=True, slots=True)
class KnowledgeEvidenceItem:
    atom_id: str
    relation: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class GetKnowledgeEvidenceResult:
    memory_space_id: str
    knowledge_id: str
    evidence: list[KnowledgeEvidenceItem]


class _SystemClock(Clock):
    """Core-compatible clock backed by UTC timestamps."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class _UuidIdentifierFactory(IdentifierFactory):
    """Core-compatible deterministic-friendly identifier factory."""

    def new_atom_id(self) -> str:
        return str(uuid.uuid4())

    def new_knowledge_id(self) -> str:
        return str(uuid.uuid4())

    def new_revision_id(self) -> str:
        return str(uuid.uuid4())

    def new_event_id(self) -> str:
        return str(uuid.uuid4())


def _to_sqlite_atom(atom: Atom, *, source_round_key: str) -> SqliteAtom:
    return SqliteAtom(
        atom_id=atom.atom_id,
        memory_space_id=atom.memory_space_id,
        source_system=atom.source.source_system,
        source_instance_id=atom.source.source_instance_id,
        source_profile_id=atom.source.source_profile_id,
        source_session_id=atom.source.source_session_id,
        source_round_id=atom.source.source_round_id,
        source_round_key=source_round_key,
        first_message_id=atom.source.first_message_id,
        final_message_id=atom.source.final_message_id,
        message_ids=atom.source.message_ids,
        source_digest=atom.source.source_digest,
        source_schema_version=atom.source.source_schema_version,
        resolver_version=atom.source.resolver_version,
        extraction_host=atom.extraction.host,
        extraction_provider=atom.extraction.provider,
        extraction_model=atom.extraction.model,
        extraction_prompt_version=atom.extraction.prompt_version,
        extraction_schema_version=atom.extraction.schema_version,
        extraction_purpose=atom.extraction.purpose,
        title=atom.content.title,
        target=atom.content.target,
        statement=atom.content.statement,
        rationale=atom.content.rationale,
        result=atom.content.result,
        artifacts=atom.content.artifacts,
        content_digest=atom.content_digest,
        supersedes_atom_id=atom.supersedes_atom_id,
        created_at=_format_timestamp(atom.created_at),
    )


def _from_sqlite_atom(row: SqliteAtom) -> Atom:
    return Atom(
        atom_id=row.atom_id,
        memory_space_id=row.memory_space_id,
        source=SourceReference(
            source_system=row.source_system,
            source_instance_id=row.source_instance_id,
            source_profile_id=row.source_profile_id,
            source_session_id=row.source_session_id,
            source_round_id=row.source_round_id,
            first_message_id=row.first_message_id,
            final_message_id=row.final_message_id,
            message_ids=row.message_ids,
            source_digest=row.source_digest,
            source_schema_version=row.source_schema_version,
            resolver_version=row.resolver_version,
        ),
        content=AtomContent(
            title=row.title,
            target=row.target,
            statement=row.statement,
            rationale=row.rationale,
            result=row.result,
            artifacts=row.artifacts,
        ),
        extraction=ExtractionInfo(
            host=row.extraction_host,
            provider=row.extraction_provider,
            model=row.extraction_model,
            prompt_version=row.extraction_prompt_version,
            schema_version=row.extraction_schema_version,
            purpose=row.extraction_purpose,
        ),
        content_digest=row.content_digest,
        created_at=_parse_timestamp(row.created_at),
        supersedes_atom_id=row.supersedes_atom_id,
    )


def _to_sqlite_knowledge(item: KnowledgeItem) -> SqliteKnowledge:
    return SqliteKnowledge(
        knowledge_id=item.knowledge_id,
        memory_space_id=item.memory_space_id,
        title=item.title,
        target=item.target,
        statement=item.statement,
        rationale=item.rationale,
        phase=item.phase.value,
        version=item.version,
        created_at=_format_timestamp(item.created_at),
        updated_at=_format_timestamp(item.updated_at),
        superseded_by_id=item.superseded_by_id,
        deleted_at=_format_timestamp(item.deleted_at)
        if item.deleted_at is not None
        else None,
    )


def _from_sqlite_knowledge(row: SqliteKnowledge, *, phase: str) -> KnowledgeItem:
    from ledgermind_core.domain import Phase

    return KnowledgeItem(
        knowledge_id=row.knowledge_id,
        memory_space_id=row.memory_space_id,
        title=row.title,
        target=row.target,
        statement=row.statement,
        rationale=row.rationale,
        phase=Phase(phase),
        version=row.version,
        created_at=_parse_timestamp(row.created_at),
        updated_at=_parse_timestamp(row.updated_at),
        superseded_by_id=row.superseded_by_id,
        deleted_at=_parse_timestamp(row.deleted_at)
        if row.deleted_at is not None
        else None,
    )


def _to_sqlite_evidence(link: KnowledgeEvidence) -> SqliteKnowledgeEvidence:
    return SqliteKnowledgeEvidence(
        knowledge_id=link.knowledge_id,
        atom_id=link.atom_id,
        relation=link.relation.value,
        created_at=_format_timestamp(link.created_at),
    )


def _to_sqlite_revision(item: KnowledgeRevision) -> SqliteKnowledgeRevision:
    return SqliteKnowledgeRevision(
        revision_id=item.revision_id,
        knowledge_id=item.knowledge_id,
        version=item.version,
        event_type=item.event_type,
        snapshot_json=item.snapshot_json,
        cause_atom_id=item.cause_atom_id,
        created_at=_format_timestamp(item.created_at),
    )


def _from_sqlite_revision(row: SqliteKnowledgeRevision) -> KnowledgeRevision:
    return KnowledgeRevision.from_snapshot(
        revision_id=RevisionId(row.revision_id),
        knowledge_id=KnowledgeId(row.knowledge_id),
        version=row.version,
        event_type=row.event_type,
        snapshot=json.loads(row.snapshot_json),
        cause_atom_id=AtomId(row.cause_atom_id) if row.cause_atom_id is not None else None,
        created_at=_parse_timestamp(row.created_at),
    )


def _to_sqlite_idempotency_result(
    result: StoredIdempotencyResult,
    *,
    created_at: datetime,
) -> SqliteStoredIdempotencyResult:
    return SqliteStoredIdempotencyResult(
        key=result.key,
        request_hash=result.request_hash,
        response_json=result.response_json,
        created_at=_format_timestamp(created_at),
        expires_at=None,
        memory_space_id=result.memory_space_id,
    )


def _to_sqlite_outbox_event(event: DomainEvent) -> SqliteOutboxEvent:
    payload = json.loads(event.payload_json)
    if not isinstance(payload, dict):
        payload = {"payload": payload}

    if not payload.get("event_type") and not payload.get("aggregate_id"):
        payload = {
            "event_type": event.event_type,
            "aggregate_id": event.aggregate_id,
        }
    event_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return SqliteOutboxEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        aggregate_id=event.aggregate_id,
        memory_space_id=event.memory_space_id,
        payload_json=event_payload,
        occurred_at=_format_timestamp(event.occurred_at),
        attempts=0,
        available_at=_format_timestamp(event.occurred_at),
        claimed_at=None,
        claimed_by=None,
        processed_at=None,
        last_error=None,
    )


class _CoreAtomRepository(AtomRepository):
    """Adapter from core atom repository contract to SQLite repository."""

    def __init__(
        self,
        delegate: SQLiteAtomRepository,
        *,
        memory_spaces: SQLiteMemorySpaceRepository,
    ) -> None:
        self._delegate = delegate
        self._memory_spaces = memory_spaces

    def get(self, memory_space_id: str, atom_id: str) -> Atom | None:
        row = self._delegate.get(atom_id=atom_id, memory_space_id=memory_space_id)
        return _from_sqlite_atom(row) if row is not None else None

    def find_by_source_version(
        self,
        memory_space_id: str,
        source_round_key: str,
        prompt_version: int,
        schema_version: int,
    ) -> Atom | None:
        row = self._delegate.get_by_source_round(
            memory_space_id=memory_space_id,
            source_round_key=source_round_key,
            extraction_prompt_version=prompt_version,
            extraction_schema_version=schema_version,
        )
        return _from_sqlite_atom(row) if row is not None else None

    def add(self, atom: Atom) -> None:
        self._memory_spaces.ensure(atom.memory_space_id, atom.source.source_system)
        self._delegate.add(_to_sqlite_atom(atom, source_round_key=calculate_source_round_key(atom.source)))


class _CoreKnowledgeRepository(KnowledgeRepository):
    """Adapter from core knowledge repository contract to SQLite repository."""

    def __init__(self, delegate: SQLiteKnowledgeRepository) -> None:
        self._delegate = delegate

    def get(self, memory_space_id: str, knowledge_id: str) -> KnowledgeItem | None:
        row = self._delegate.get(knowledge_id=knowledge_id, memory_space_id=memory_space_id)
        return _from_sqlite_knowledge(row, phase=row.phase) if row is not None else None

    def add(self, item: KnowledgeItem) -> None:
        self._delegate.add(_to_sqlite_knowledge(item))

    def update(self, item: KnowledgeItem, expected_version: int) -> None:
        self._delegate.update(_to_sqlite_knowledge(item), expected_version=expected_version)

    def get_many(
        self,
        memory_space_id: str,
        knowledge_ids: tuple[str, ...],
    ) -> list[KnowledgeItem]:
        if not knowledge_ids:
            return []

        items = self._delegate.list_by_space(memory_space_id)
        lookup = {
            item.knowledge_id: _from_sqlite_knowledge(item, phase=item.phase)
            for item in items
        }
        return [lookup[item_id] for item_id in knowledge_ids if item_id in lookup]


class _CoreEvidenceRepository(EvidenceRepository):
    """Adapter from core evidence repository contract to SQLite repository."""

    def __init__(self, delegate: SQLiteEvidenceRepository) -> None:
        self._delegate = delegate

    def add(self, link: KnowledgeEvidence) -> None:
        self._delegate.add(_to_sqlite_evidence(link))

    def count_for_knowledge(self, memory_space_id: str, knowledge_id: str) -> int:
        return self._delegate.count_for_knowledge(memory_space_id, knowledge_id)

    def list_atom_ids(self, memory_space_id: str, knowledge_id: str) -> list[str]:
        return self._delegate.list_atom_ids(memory_space_id, knowledge_id)

    def list_for_knowledge(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> list[KnowledgeEvidence]:
        return cast(list[KnowledgeEvidence], self._delegate.list_for_knowledge(memory_space_id, knowledge_id))

    def list_for_atom(
        self,
        memory_space_id: str,
        atom_id: str,
    ) -> list[KnowledgeEvidence]:
        return cast(list[KnowledgeEvidence], self._delegate.list_for_atom(memory_space_id, atom_id))


class _CoreRevisionRepository(RevisionRepository):
    """Adapter from core revision repository contract to SQLite repository."""

    def __init__(self, delegate: SQLiteRevisionRepository) -> None:
        self._delegate = delegate

    def add(self, item: KnowledgeRevision) -> None:
        self._delegate.add(_to_sqlite_revision(item))

    def list_for_knowledge(
        self,
        memory_space_id: str,
        knowledge_id: str,
    ) -> list[KnowledgeRevision]:
        rows = self._delegate.list_for_knowledge(memory_space_id, knowledge_id)
        return [_from_sqlite_revision(row) for row in rows]


class GetKnowledgeHistoryHandler:
    """Build history timeline for one knowledge item."""

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def handle(
        self,
        query: GetKnowledgeHistoryQuery,
    ) -> GetKnowledgeHistoryResult | None:
        with self._uow_factory() as uow:
            knowledge = uow.knowledge.get(query.memory_space_id, query.knowledge_id)
            if knowledge is None:
                return None

            revisions = [
                KnowledgeHistoryItem(
                    revision_id=revision.revision_id,
                    version=revision.version,
                    event_type=revision.event_type,
                    snapshot=revision.snapshot,
                    cause_atom_id=revision.cause_atom_id,
                    created_at=_format_timestamp(revision.created_at),
                )
                for revision in uow.revisions.list_for_knowledge(
                    query.memory_space_id,
                    query.knowledge_id,
                )
            ]

            return GetKnowledgeHistoryResult(
                memory_space_id=query.memory_space_id,
                knowledge_id=query.knowledge_id,
                revisions=revisions,
            )


class GetKnowledgeEvidenceHandler:
    """Build evidence list for one knowledge item."""

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def handle(
        self,
        query: GetKnowledgeEvidenceQuery,
    ) -> GetKnowledgeEvidenceResult | None:
        with self._uow_factory() as uow:
            knowledge = uow.knowledge.get(query.memory_space_id, query.knowledge_id)
            if knowledge is None:
                return None

            evidence = [
                KnowledgeEvidenceItem(
                    atom_id=item.atom_id,
                    relation=str(item.relation),
                    created_at=item.created_at,
                )
                for item in uow.evidence.list_for_knowledge(
                    query.memory_space_id,
                    query.knowledge_id,
                )
            ]

            return GetKnowledgeEvidenceResult(
                memory_space_id=query.memory_space_id,
                knowledge_id=query.knowledge_id,
                evidence=evidence,
            )


class _CoreIdempotencyRepository(IdempotencyRepository):
    """Adapter from core idempotency repository contract to SQLite repository."""

    def __init__(self, delegate: SQLiteIdempotencyRepository, *, clock: Clock) -> None:
        self._delegate = delegate
        self._clock = clock

    def get(self, memory_space_id: str, key: str) -> StoredIdempotencyResult | None:
        row = self._delegate.get(memory_space_id, key)
        if row is None:
            return None
        return StoredIdempotencyResult(
            key=row.key,
            request_hash=row.request_hash,
            response_json=row.response_json,
            memory_space_id=row.memory_space_id,
        )

    def add(self, result: StoredIdempotencyResult) -> None:
        self._delegate.add(
            _to_sqlite_idempotency_result(
                result,
                created_at=self._clock.now(),
            )
        )


class _CoreOutboxEventRepository(EventRepository):
    """Adapter from core event repository contract to SQLite durable outbox."""

    def __init__(
        self,
        delegate: SQLiteOutboxRepository,
        projection_names: tuple[str, ...] = DEFAULT_PROJECTIONS,
    ) -> None:
        self._delegate = delegate
        self._projection_names = projection_names

    def add(self, event: DomainEvent) -> None:
        row = _to_sqlite_outbox_event(event)
        self._delegate.add(row, projection_names=self._projection_names)

    def committed(self) -> tuple[DomainEvent, ...]:
        return ()

    @property
    def stored_events(self) -> tuple[DomainEvent, ...]:
        return ()


class _SQLiteCoreUnitOfWork(UnitOfWork):
    """Core UnitOfWork adapter around :class:`persistence.SQLiteUnitOfWork`."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        write_transaction: bool = True,
        clock: Clock | None = None,
        identifiers: IdentifierFactory | None = None,
        projection_names: tuple[str, ...] = DEFAULT_PROJECTIONS,
    ) -> None:
        self._database_path = str(database_path)
        self._busy_timeout_ms = busy_timeout_ms
        self._write_transaction = write_transaction
        self._clock = clock or _SystemClock()
        self._identifiers = identifiers or _UuidIdentifierFactory()
        self._projection_names = projection_names
        self._uow: SQLiteUnitOfWork | None = None
        self._atoms: _CoreAtomRepository | None = None
        self._knowledge: _CoreKnowledgeRepository | None = None
        self._evidence: _CoreEvidenceRepository | None = None
        self._revisions: _CoreRevisionRepository | None = None
        self._idempotency: _CoreIdempotencyRepository | None = None
        self._events: _CoreOutboxEventRepository | None = None
        self._search: KnowledgeSearch | None = None

    def __enter__(self) -> _SQLiteCoreUnitOfWork:  # noqa: PYI034
        if self._uow is not None:
            return self

        uow = SQLiteUnitOfWork(
            database_path=self._database_path,
            busy_timeout_ms=self._busy_timeout_ms,
            write_transaction=self._write_transaction,
        )
        uow.__enter__()

        self._uow = uow
        self._atoms = _CoreAtomRepository(
            delegate=uow.atoms,
            memory_spaces=uow.memory_spaces,
        )
        self._knowledge = _CoreKnowledgeRepository(uow.knowledge)
        self._evidence = _CoreEvidenceRepository(uow.evidence)
        self._revisions = _CoreRevisionRepository(uow.revisions)
        self._idempotency = _CoreIdempotencyRepository(
            delegate=uow.idempotency,
            clock=self._clock,
        )
        self._events = _CoreOutboxEventRepository(
            uow.outbox,
            projection_names=self._projection_names,
        )
        self._search = _build_search_adapter(
            connection=uow.connection,
            database_path=self._database_path,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._uow is not None:
            self._uow.__exit__(exc_type, exc, tb)
        self._reset_state()

    def _reset_state(self) -> None:
        self._uow = None
        self._atoms = None
        self._knowledge = None
        self._evidence = None
        self._revisions = None
        self._idempotency = None
        self._events = None
        self._search = None

    @property
    def atoms(self) -> _CoreAtomRepository:
        if self._atoms is None:
            raise RuntimeError("unit of work is not active")
        return self._atoms

    @atoms.setter
    def atoms(self, value: _CoreAtomRepository) -> None:
        self._atoms = value

    @property
    def knowledge(self) -> _CoreKnowledgeRepository:
        if self._knowledge is None:
            raise RuntimeError("unit of work is not active")
        return self._knowledge

    @knowledge.setter
    def knowledge(self, value: _CoreKnowledgeRepository) -> None:
        self._knowledge = value

    @property
    def evidence(self) -> _CoreEvidenceRepository:
        if self._evidence is None:
            raise RuntimeError("unit of work is not active")
        return self._evidence

    @evidence.setter
    def evidence(self, value: _CoreEvidenceRepository) -> None:
        self._evidence = value

    @property
    def revisions(self) -> _CoreRevisionRepository:
        if self._revisions is None:
            raise RuntimeError("unit of work is not active")
        return self._revisions

    @revisions.setter
    def revisions(self, value: _CoreRevisionRepository) -> None:
        self._revisions = value

    @property
    def idempotency(self) -> _CoreIdempotencyRepository:
        if self._idempotency is None:
            raise RuntimeError("unit of work is not active")
        return self._idempotency

    @idempotency.setter
    def idempotency(self, value: _CoreIdempotencyRepository) -> None:
        self._idempotency = value

    @property
    def events(self) -> _CoreOutboxEventRepository:
        if self._events is None:
            raise RuntimeError("unit of work is not active")
        return self._events

    @events.setter
    def events(self, value: _CoreOutboxEventRepository) -> None:
        self._events = value

    @property
    def search(self) -> KnowledgeSearch:
        if self._search is None:
            raise RuntimeError("unit of work is not active")
        return self._search

    @search.setter
    def search(self, value: KnowledgeSearch) -> None:
        self._search = value

    @property
    def clock(self) -> Clock:
        return self._clock

    @clock.setter
    def clock(self, value: Clock) -> None:
        self._clock = value

    @property
    def identifiers(self) -> IdentifierFactory:
        return self._identifiers

    @identifiers.setter
    def identifiers(self, value: IdentifierFactory) -> None:
        self._identifiers = value

    def commit(self) -> None:
        if self._uow is None:
            raise RuntimeError("unit of work is not active")
        self._uow.commit()

    def rollback(self) -> None:
        if self._uow is None:
            return
        self._uow.rollback()


def build_ingest_atom_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
    projection_names: tuple[str, ...] = DEFAULT_PROJECTIONS,
) -> IngestAtomHandler:
    """Build a core-compatible ingest atom handler backed by local SQLite."""

    return IngestAtomHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            projection_names=projection_names,
        ),
        policy=IsolatedPatternPolicy(),
        clock=_SystemClock(),
        identifiers=_UuidIdentifierFactory(),
        serializer=JsonIngestAtomResultSerializer(),
    )


def build_get_atom_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
) -> GetAtomHandler:
    """Build a core-compatible atom read handler backed by local SQLite."""

    return GetAtomHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=False,
        ),
    )


def build_get_knowledge_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
) -> GetKnowledgeHandler:
    """Build a core-compatible knowledge read handler backed by local SQLite."""

    return GetKnowledgeHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=False,
        ),
    )


def build_get_knowledge_history_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
) -> GetKnowledgeHistoryHandler:
    """Build a handler returning revision history for one knowledge item."""

    return GetKnowledgeHistoryHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=False,
        ),
    )


def build_get_knowledge_evidence_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
) -> GetKnowledgeEvidenceHandler:
    """Build a handler returning evidence links for one knowledge item."""

    return GetKnowledgeEvidenceHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=False,
        ),
    )


def build_retrieve_context_handler(
    *,
    database_path: str | Path,
    busy_timeout_ms: int = 5_000,
) -> RetrieveContextHandler:
    """Build a context retrieval handler backed by local SQLite."""

    return RetrieveContextHandler(
        uow_factory=_build_uow_factory(
            database_path=database_path,
            busy_timeout_ms=busy_timeout_ms,
            write_transaction=False,
        )
    )


def bootstrap_local_service(
    *,
    home: str | Path = "~/.ledgermind/local",
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig]:
    """Prepare filesystem and return resolved runtime objects."""

    paths = ServicePaths(home=home)
    cfg = config or LocalConfig(config_version=1)
    resolved_home = paths.home
    resolved_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(paths.database_file)
    connection.close()
    return paths, cfg


def _atomic_write_text(path: Path, data: str, *, mode: int) -> None:
    """Write text file atomically and with explicit permissions."""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
            temp_for_permissions = tmp_path
            os.chmod(temp_for_permissions, mode)
            tmp_path.replace(path)
            # Avoid race with subsequent operations that only inspect perms.
            os.chmod(path, mode)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def initialize_local_layout(
    *,
    home: str | Path = "~/.ledgermind/local",
    force: bool = False,
    rotate_token: bool = False,
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig, str]:
    """Initialize local service data layout.

    - creates directory tree
    - creates or preserves config
    - creates or rotates api token
    """

    paths, cfg = bootstrap_local_service(home=home, config=config)

    config_path = paths.config_file
    if force or not config_path.exists():
        cfg = config or LocalConfig(config_version=1)
        _atomic_write_text(config_path, cfg.to_json(), mode=0o600)
    elif config is None:
        cfg = LocalConfig.from_file(config_path)
    else:
        cfg = config

    if not paths.token_file.exists() or (force and rotate_token):
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    if rotate_token:
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    existing_token = paths.token_file.read_text(encoding="utf-8")
    return paths, cfg, existing_token
