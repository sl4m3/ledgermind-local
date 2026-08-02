"""Import legacy v3 records into the canonical local SQLite schema."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ledgermind_core.application.digests import (
    calculate_atom_content_digest,
    calculate_idempotency_key,
    calculate_request_hash,
    calculate_source_round_key,
)
from ledgermind_core.domain import AtomContent, ExtractionInfo, SourceReference
from ledgermind_core.domain.events import AtomCreated, KnowledgeCreated

from ledgermind_local.persistence import SQLiteUnitOfWork, migrations
from ledgermind_local.persistence.atom_repository import Atom
from ledgermind_local.persistence.evidence_repository import KnowledgeEvidence
from ledgermind_local.persistence.idempotency_repository import StoredIdempotencyResult
from ledgermind_local.persistence.knowledge_repository import Knowledge
from ledgermind_local.persistence.outbox_repository import OutboxEvent
from ledgermind_local.persistence.revision_repository import KnowledgeRevision
from ledgermind_local.v3_migration.models import LegacyRecord, MigrationManifest

_DEFAULT_PROJECTION_NAMES = (
    "projections.search",
    "projections.knowledge",
    "projections.markdown",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_record(
    mapped: Mapping[str, Any],
) -> tuple[SourceReference, AtomContent, ExtractionInfo]:
    raw_digest = str(mapped["source_digest"])
    source_digest = raw_digest if raw_digest.startswith("sha256:") else f"sha256:{raw_digest}"
    source = SourceReference(
        source_system=str(mapped["source_system"]),
        source_instance_id=str(mapped["source_instance_id"]),
        source_profile_id=str(mapped["source_profile_id"]),
        source_session_id=str(mapped["source_session_id"]),
        source_round_id=str(mapped["source_round_id"]),
        first_message_id=None,
        final_message_id=None,
        message_ids=(),
        source_digest=source_digest,
        source_schema_version=int(mapped["schema_version"]),
        resolver_version=1,
    )
    content = AtomContent(
        title=str(mapped.get("title") or ""),
        target=str(mapped.get("target") or ""),
        statement=str(mapped.get("statement") or ""),
        rationale=str(mapped.get("rationale") or ""),
        result="",
        artifacts=tuple(str(item) for item in (mapped.get("artifacts") or [])),
    )
    extraction = ExtractionInfo(
        host=str(mapped["extraction_host"]),
        provider=str(mapped["extraction_provider"]),
        model=str(mapped.get("extraction_model") or ""),
        prompt_version=int(mapped["prompt_version"]),
        schema_version=int(mapped["schema_version"]),
        purpose="ledgermind.atom.extract",
    )
    return source, content, extraction


def _request_payload(
    *,
    memory_space_id: str,
    source: SourceReference,
    content: AtomContent,
    extraction: ExtractionInfo,
) -> dict[str, object]:
    return {
        "api_version": "1",
        "memory_space_id": memory_space_id,
        "source": {
            "source_system": source.source_system,
            "source_instance_id": source.source_instance_id,
            "source_profile_id": source.source_profile_id,
            "source_session_id": source.source_session_id,
            "source_round_id": source.source_round_id,
            "first_message_id": source.first_message_id,
            "final_message_id": source.final_message_id,
            "message_ids": list(source.message_ids),
            "source_digest": source.source_digest,
            "source_schema_version": source.source_schema_version,
            "resolver_version": source.resolver_version,
        },
        "atom": {
            "title": content.title,
            "target": content.target,
            "statement": content.statement,
            "rationale": content.rationale,
            "result": content.result,
            "artifacts": list(content.artifacts),
        },
        "extraction": {
            "host": extraction.host,
            "provider": extraction.provider,
            "model": extraction.model,
            "prompt_version": extraction.prompt_version,
            "schema_version": extraction.schema_version,
            "purpose": extraction.purpose,
        },
    }


def _event_payload(event_type: str, aggregate_id: str) -> str:
    return json.dumps(
        {"event_type": event_type, "aggregate_id": aggregate_id},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _add_canonical_record(
    *,
    uow: SQLiteUnitOfWork,
    record: LegacyRecord,
    mapped: Mapping[str, Any],
    migration_id: str,
) -> tuple[str, str, str, str | None]:
    source, content, extraction = _canonical_record(mapped)
    memory_space_id = str(mapped["memory_space_id"])
    source_round_key = calculate_source_round_key(source)
    existing = uow.atoms.get_by_source_round(
        memory_space_id=memory_space_id,
        source_round_key=source_round_key,
        extraction_prompt_version=extraction.prompt_version,
        extraction_schema_version=extraction.schema_version,
    )
    if existing is not None:
        evidence = uow.evidence.list_for_atom(memory_space_id, existing.atom_id)
        knowledge_id = evidence[0].knowledge_id if evidence else ""
        return existing.atom_id, knowledge_id, "duplicate", "existing_source_round"

    atom_id = f"legacy:{migration_id}:{record.fid}"
    knowledge_id = f"legacy:{migration_id}:{record.fid}:knowledge"
    now = _now_iso()
    phase = str(mapped["phase"]).lower()
    content_digest = calculate_atom_content_digest(
        content=content,
        source=source,
        extraction=extraction,
    )

    uow.memory_spaces.ensure(memory_space_id, source.source_system)
    uow.atoms.add(
        Atom(
            atom_id=atom_id,
            memory_space_id=memory_space_id,
            source_system=source.source_system,
            source_instance_id=source.source_instance_id,
            source_profile_id=source.source_profile_id,
            source_session_id=source.source_session_id,
            source_round_id=source.source_round_id,
            source_round_key=source_round_key,
            first_message_id=source.first_message_id,
            final_message_id=source.final_message_id,
            message_ids=source.message_ids,
            source_digest=source.source_digest,
            source_schema_version=source.source_schema_version,
            resolver_version=source.resolver_version,
            extraction_host=extraction.host,
            extraction_provider=extraction.provider,
            extraction_model=extraction.model,
            extraction_prompt_version=extraction.prompt_version,
            extraction_schema_version=extraction.schema_version,
            extraction_purpose=extraction.purpose,
            title=content.title,
            target=content.target,
            statement=content.statement,
            rationale=content.rationale,
            result=content.result,
            artifacts=content.artifacts,
            content_digest=content_digest,
            supersedes_atom_id=None,
            created_at=now,
        )
    )
    uow.knowledge.add(
        Knowledge(
            knowledge_id=knowledge_id,
            memory_space_id=memory_space_id,
            title=content.title,
            target=content.target,
            statement=content.statement,
            rationale=content.rationale,
            phase=phase,
            version=1,
            created_at=now,
            updated_at=now,
            superseded_by_id=None,
            deleted_at=None,
        )
    )
    uow.evidence.add(
        KnowledgeEvidence(
            knowledge_id=knowledge_id,
            atom_id=atom_id,
            relation="origin",
            created_at=now,
        )
    )
    snapshot = {
        "knowledge_id": knowledge_id,
        "memory_space_id": memory_space_id,
        "title": content.title,
        "target": content.target,
        "statement": content.statement,
        "rationale": content.rationale,
        "phase": phase,
        "version": 1,
        "superseded_by_id": None,
        "deleted_at": None,
    }
    uow.revisions.add(
        KnowledgeRevision(
            revision_id=f"legacy:revision:{migration_id}:{record.fid}",
            knowledge_id=knowledge_id,
            version=1,
            event_type=KnowledgeCreated.EVENT_NAME,
            snapshot_json=json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            cause_atom_id=atom_id,
            created_at=now,
        )
    )

    for event_id, event_type, aggregate_id in (
        (f"legacy:event:{migration_id}:{record.fid}:atom", AtomCreated.EVENT_NAME, atom_id),
        (f"legacy:event:{migration_id}:{record.fid}:knowledge", KnowledgeCreated.EVENT_NAME, knowledge_id),
    ):
        uow.outbox.add(
            OutboxEvent(
                event_id=event_id,
                event_type=event_type,
                aggregate_id=aggregate_id,
                memory_space_id=memory_space_id,
                payload_json=_event_payload(event_type, aggregate_id),
                occurred_at=now,
                available_at=now,
                attempts=0,
                claimed_at=None,
                claimed_by=None,
                processed_at=None,
                last_error=None,
            ),
            projection_names=_DEFAULT_PROJECTION_NAMES,
        )

    request = _request_payload(
        memory_space_id=memory_space_id,
        source=source,
        content=content,
        extraction=extraction,
    )
    idempotency_key = calculate_idempotency_key(
        source_round_key=source_round_key,
        extraction_prompt_version=extraction.prompt_version,
        extraction_schema_version=extraction.schema_version,
    )
    response_json = json.dumps(
        {
            "atom_id": atom_id,
            "knowledge_id": knowledge_id,
            "knowledge_version": 1,
            "phase": phase,
            "duplicate": False,
            "projections_pending": True,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    uow.idempotency.add(
        StoredIdempotencyResult(
            key=idempotency_key,
            request_hash=calculate_request_hash(request),
            response_json=response_json,
            created_at=now,
            expires_at=None,
            memory_space_id=memory_space_id,
        )
    )
    return atom_id, knowledge_id, "migrated", None


def write_temp_migration(
    *,
    records: list[LegacyRecord],
    destination: Path,
    migration_id: str,
    apply: bool,
) -> tuple[list[MigrationManifest], list[str]]:
    """Import v3 records into the canonical local SQLite schema.

    Dry-run uses a savepoint and commits only the canonical schema, while the
    record writes are rolled back. Apply mode commits the same writes.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifests: list[MigrationManifest] = []
    warnings: list[str] = []
    with SQLiteUnitOfWork(destination) as uow:
        migrations.apply_migrations(uow.connection)
        if not apply:
            uow.connection.execute("SAVEPOINT v3_migration_preview")

        for record in records:
            try:
                from ledgermind_local.v3_migration.mapper import map_record

                mapped = map_record(record)
            except Exception as exc:  # noqa: BLE001
                map_warning = f"map_failed:{record.fid}:{exc}"
                warnings.append(map_warning)
                manifests.append(
                    MigrationManifest(
                        migration_id=migration_id,
                        legacy_fid=record.fid,
                        action="skipped",
                        warnings_json=json.dumps([map_warning], ensure_ascii=False),
                    )
                )
                continue

            try:
                atom_id, knowledge_id, action, warning = _add_canonical_record(
                    uow=uow,
                    record=record,
                    mapped=mapped,
                    migration_id=migration_id,
                )
            except (TypeError, ValueError) as exc:
                validation_warning = f"canonical_validation_failed:{record.fid}:{exc}"
                warnings.append(validation_warning)
                manifests.append(
                    MigrationManifest(
                        migration_id=migration_id,
                        legacy_fid=record.fid,
                        action="skipped",
                        warnings_json=json.dumps([validation_warning], ensure_ascii=False),
                    )
                )
                continue
            item_warnings = list(mapped.get("warnings", []))
            if warning:
                item_warnings.append(warning)
            warnings.extend(item_warnings)
            manifests.append(
                MigrationManifest(
                    migration_id=migration_id,
                    legacy_fid=record.fid,
                    atom_id=atom_id,
                    knowledge_id=knowledge_id,
                    action="planned" if not apply and action == "migrated" else action,
                    warnings_json=json.dumps(item_warnings, ensure_ascii=False),
                )
            )

        if not apply:
            uow.connection.execute("ROLLBACK TO v3_migration_preview")
            uow.connection.execute("RELEASE v3_migration_preview")
        uow.commit()

    return manifests, warnings