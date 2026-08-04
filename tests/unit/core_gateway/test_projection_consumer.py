from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import pytest

from ledgermind_local.core_gateway.projection_consumer import CoreProjectionConsumer
from ledgermind_local.core_gateway.projection_contracts import (
    CORE_PROJECTION_UPSERT,
    CoreProjectionEvent,
    PollProjectionEventsResult,
)
from ledgermind_local.core_gateway.projection_inbox import CoreProjectionInbox
from ledgermind_local.persistence import rounds_migrations


@dataclass
class FakeGateway:
    event: CoreProjectionEvent
    calls: list[str]

    def poll_projection_events(self, command):
        self.calls.append("poll")
        return PollProjectionEventsResult(events=(self.event,), has_more=False)

    def ack_projection_events(self, command):
        self.calls.append("ack")
        return type("Ack", (), {"acknowledged": command.event_ids})()


class Handler:
    def __init__(self, calls: list[str], fail_once: bool = False) -> None:
        self.calls = calls
        self.fail_once = fail_once

    def handle_core_event(self, event: CoreProjectionEvent) -> None:
        self.calls.append(event.event_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("projection backend unavailable")


def event() -> CoreProjectionEvent:
    return CoreProjectionEvent(
        event_id="event-1",
        memory_space_id="space-1",
        aggregate_id="knowledge-1",
        event_type=CORE_PROJECTION_UPSERT,
        payload_json=json.dumps(
            {
                "knowledge_id": "knowledge-1",
                "memory_space_id": "space-1",
                "title": "Title",
                "target": "Target",
                "statement": "Statement",
                "projection_version": 1,
            },
            separators=(",", ":"),
        ),
        occurred_at="2026-01-01T00:00:00Z",
    )


def inbox() -> tuple[sqlite3.Connection, CoreProjectionInbox]:
    connection = sqlite3.connect(":memory:")
    rounds_migrations.apply_migrations(connection)
    return connection, CoreProjectionInbox(connection)


def test_consumer_persists_before_ack_and_does_not_reapply_processed_event() -> None:
    connection, inbox_store = inbox()
    gateway_calls: list[str] = []
    handler_calls: list[str] = []
    consumer = CoreProjectionConsumer(
        gateway=FakeGateway(event(), gateway_calls),
        inbox=inbox_store,
        handlers={"fts": Handler(handler_calls)},
    )

    first = consumer.poll_once("space-1", "local-projections")
    assert first.persisted == 1
    assert first.acknowledged == 1
    assert first.processed == 1
    assert gateway_calls == ["poll", "ack"]
    assert handler_calls == ["event-1"]

    second = consumer.poll_once("space-1", "local-projections")
    assert second.processed == 0
    assert handler_calls == ["event-1"]
    connection.close()


def test_failed_projection_is_retained_for_retry_after_core_ack() -> None:
    connection, inbox_store = inbox()
    gateway_calls: list[str] = []
    handler_calls: list[str] = []
    handler = Handler(handler_calls, fail_once=True)
    consumer = CoreProjectionConsumer(
        gateway=FakeGateway(event(), gateway_calls),
        inbox=inbox_store,
        handlers={"fts": handler},
    )

    first = consumer.poll_once("space-1", "local-projections")
    assert first.acknowledged == 1
    assert first.failed == 1
    assert handler_calls == ["event-1"]

    retry = consumer.process_pending()
    assert retry.processed == 1
    assert retry.failed == 0
    assert handler_calls == ["event-1", "event-1"]
    connection.close()


def test_consumer_rejects_handler_without_core_event_entrypoint() -> None:
    connection, inbox_store = inbox()
    with pytest.raises(TypeError, match="handle_core_event"):
        CoreProjectionConsumer(
            gateway=FakeGateway(event(), []),
            inbox=inbox_store,
            handlers={"fts": object()},
        )
    connection.close()
