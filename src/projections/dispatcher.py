"""Projection delivery dispatcher for durable outbox events."""

from __future__ import annotations

from typing import Mapping, Protocol

from persistence import OutboxEvent


class _ProjectionHandler(Protocol):
    """Minimal projection handler contract used by the dispatcher."""

    def handle_event(
        self,
        *,
        event_type: str,
        memory_space_id: str,
        aggregate_id: str,
        payload_json: str,
    ) -> bool: ...


class ProjectionDispatcher:
    """Dispatch `OutboxEvent` objects to named projection handlers."""

    def __init__(self, handlers: Mapping[str, _ProjectionHandler]):
        self._handlers = dict(handlers)

    @property
    def projection_names(self) -> tuple[str, ...]:
        return tuple(self._handlers.keys())

    def dispatch(self, projection_name: str, event: OutboxEvent) -> bool:
        """Invoke one projection handler for the event and return handler result."""

        if projection_name not in self._handlers:
            raise KeyError(f"unknown projection: {projection_name}")

        handler = self._handlers[projection_name]
        return bool(
            handler.handle_event(
                event_type=event.event_type,
                memory_space_id=event.memory_space_id,
                aggregate_id=event.aggregate_id,
                payload_json=event.payload_json,
            )
        )

