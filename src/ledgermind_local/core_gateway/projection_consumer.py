"""Consume Core projection events without opening the Core database."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base import CoreGateway
from .projection_contracts import (
    AckProjectionEventsCommand,
    PollProjectionEventsCommand,
)
from .projection_inbox import CoreProjectionInbox


@dataclass(frozen=True, slots=True)
class ProjectionConsumerStats:
    fetched: int = 0
    persisted: int = 0
    acknowledged: int = 0
    processed: int = 0
    failed: int = 0


class CoreProjectionConsumer:
    """Persist, acknowledge, and apply Core events with retryable Local delivery."""

    def __init__(
        self,
        *,
        gateway: CoreGateway,
        inbox: CoreProjectionInbox,
        handlers: Mapping[str, Any],
    ) -> None:
        if not handlers:
            raise ValueError("at least one Core projection handler is required")
        for name, handler in handlers.items():
            if not name or not callable(getattr(handler, "handle_core_event", None)):
                raise TypeError(f"projection handler {name!r} must define handle_core_event")
        self._gateway = gateway
        self._inbox = inbox
        self._handlers = dict(handlers)

    def poll_once(
        self,
        memory_space_id: str,
        consumer_id: str,
        *,
        limit: int = 100,
    ) -> ProjectionConsumerStats:
        result = self._gateway.poll_projection_events(
            PollProjectionEventsCommand(
                request_id=f"projection-poll-{uuid.uuid4().hex}",
                memory_space_id=memory_space_id,
                consumer_id=consumer_id,
                limit=limit,
            )
        )
        events = tuple(result.events)
        self._inbox.save_events(events, projection_names=self._handlers)
        acknowledged = 0
        if events:
            ack = self._gateway.ack_projection_events(
                AckProjectionEventsCommand(
                    request_id=f"projection-ack-{uuid.uuid4().hex}",
                    consumer_id=consumer_id,
                    event_ids=tuple(event.event_id for event in events),
                )
            )
            acknowledged = len(ack.acknowledged)
        applied = self.process_pending()
        return ProjectionConsumerStats(
            fetched=len(events),
            persisted=len(events),
            acknowledged=acknowledged,
            processed=applied.processed,
            failed=applied.failed,
        )

    def process_pending(self, *, limit: int = 100) -> ProjectionConsumerStats:
        processed = 0
        failed = 0
        for projection_name, handler in self._handlers.items():
            for delivery in self._inbox.ready(projection_name, limit=limit):
                try:
                    handler.handle_core_event(delivery.event)
                except Exception as exc:  # noqa: BLE001 - failed delivery is persisted for retry
                    self._inbox.mark_failed(
                        projection_name,
                        delivery.event_id,
                        f"{type(exc).__name__} while applying projection",
                    )
                    failed += 1
                else:
                    self._inbox.mark_processed(projection_name, delivery.event_id)
                    processed += 1
        return ProjectionConsumerStats(processed=processed, failed=failed)


__all__ = ["CoreProjectionConsumer", "ProjectionConsumerStats"]
