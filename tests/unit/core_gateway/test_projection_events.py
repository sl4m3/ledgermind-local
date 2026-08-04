from __future__ import annotations

import json

import pytest

from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_DELETE,
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    ProjectionDeletePayload,
    ProjectionUpsertPayload,
)


def _upsert_payload() -> dict[str, object]:
    return {
        "knowledge_id": "knowledge-1",
        "memory_space_id": "space-a",
        "title": "Title",
        "target": "target",
        "statement": "Statement",
        "projection_version": 1,
    }


def test_core_projection_upsert_payload_is_public_only() -> None:
    payload = ProjectionUpsertPayload.from_payload(_upsert_payload())

    assert payload.to_payload() == _upsert_payload()
    with pytest.raises(ValueError, match="unknown projection payload fields"):
        ProjectionUpsertPayload.from_payload(
            {**_upsert_payload(), "phase": "canonical"}
        )


def test_core_projection_delete_payload_is_strict() -> None:
    payload = ProjectionDeletePayload.from_payload(
        {
            "knowledge_id": "knowledge-1",
            "memory_space_id": "space-a",
            "projection_version": 2,
        }
    )

    assert payload.to_payload()["knowledge_id"] == "knowledge-1"
    with pytest.raises(ValueError, match="unknown projection payload fields"):
        ProjectionDeletePayload.from_payload(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-a",
                "projection_version": 2,
                "evidence_count": 4,
            }
        )


def test_projection_event_validates_event_type_and_memory_space() -> None:
    event = CoreProjectionEvent(
        event_id="event-1",
        memory_space_id="space-a",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(_upsert_payload()),
        occurred_at="2026-08-03T00:00:00+00:00",
    )

    assert event.parse_payload().knowledge_id == "knowledge-1"

    mismatched = CoreProjectionEvent(
        event_id=event.event_id,
        memory_space_id=event.memory_space_id,
        aggregate_id=event.aggregate_id,
        event_type=CORE_PROJECTION_DELETE,
        payload_json=event.payload_json,
        occurred_at=event.occurred_at,
    )
    with pytest.raises(ValueError, match="does not match event type"):
        mismatched.parse_payload()
