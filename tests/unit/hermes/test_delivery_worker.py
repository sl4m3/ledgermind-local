"""Tests for Hermes delivery worker retry and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

from ledgermind_local.plugins.hermes.client import (
    LedgerMindConflictError,
    LedgerMindNetworkError,
    LedgerMindUnauthorizedError,
)
from ledgermind_local.plugins.hermes.delivery_worker import DeliveryWorker
from ledgermind_local.plugins.hermes.spool import FileSpool


class _FakeClient:
    def __init__(self, exception: Exception | None = None):
        self.exception = exception
        self.calls = 0
        self.payloads: list[dict[str, object]] = []

    def ingest_atom(self, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
        del timeout
        self.calls += 1
        self.payloads.append(payload)
        if self.exception is None:
            return payload
        raise self.exception


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_delivery_worker_moves_conflicts_to_failed(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "conflict-item"
    payload = {"memory_space_id": "space-a", "idempotency_key": "key-1"}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(exception=LedgerMindConflictError("duplicate")),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert not (spool.ready_dir / f"{item_name}.json").exists()
    assert (spool.failed_dir / f"{item_name}.json").exists()

    failed = _read_json(spool.failed_dir / f"{item_name}.json")
    assert failed["delivery"]["failure_reason"].startswith("idempotency_conflict:")


def test_delivery_worker_keeps_ready_item_on_network_error(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "network-item"
    payload = {"memory_space_id": "space-a", "idempotency_key": "key-2"}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(exception=LedgerMindNetworkError("offline")),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert (spool.ready_dir / f"{item_name}.json").exists()
    assert not (spool.failed_dir / f"{item_name}.json").exists()
    ready = _read_json(spool.ready_dir / f"{item_name}.json")
    assert ready["delivery"]["attempts"] == 1
    assert ready["request"] == payload


def test_delivery_worker_retries_on_unauthorized_once(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "unauthorized-item"
    payload = {"memory_space_id": "space-a", "idempotency_key": "key-3"}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(exception=LedgerMindUnauthorizedError("expired")),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert (spool.ready_dir / f"{item_name}.json").exists()
    ready = _read_json(spool.ready_dir / f"{item_name}.json")
    assert ready["delivery"]["attempts"] == 1


def test_delivery_worker_removes_item_after_success(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "success-item"
    payload = {"memory_space_id": "space-a", "idempotency_key": "key-4"}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert not (spool.ready_dir / f"{item_name}.json").exists()
