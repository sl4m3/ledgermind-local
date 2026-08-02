"""Tests for Hermes delivery worker retry and failure handling."""

from __future__ import annotations

from pathlib import Path
import json

from plugins.hermes.client import (
    LedgerMindConflictError,
    LedgerMindNetworkError,
    LedgerMindUnauthorizedError,
)
from plugins.hermes.delivery_worker import DeliveryWorker
from plugins.hermes.spool import FileSpool


class _FakeClient:
    def __init__(self, exception: Exception | None = None):
        self.exception = exception
        self.calls = 0

    def ingest_atom(self, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
        del timeout
        self.calls += 1
        if self.exception is None:
            return payload
        raise self.exception


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_delivery_worker_moves_conflicts_to_failed(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "conflict-item"
    payload = {"id": 1}
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
    payload = {"id": 2}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(exception=LedgerMindNetworkError("offline")),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert (spool.ready_dir / f"{item_name}.json").exists()
    assert not (spool.failed_dir / f"{item_name}.json").exists()
    assert payload == _read_json(spool.ready_dir / f"{item_name}.json")


def test_delivery_worker_retries_on_unauthorized_once(tmp_path: Path) -> None:
    spool = FileSpool(tmp_path / "spool")
    item_name = "unauthorized-item"
    payload = {"id": 3}
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
    payload = {"id": 4}
    spool.enqueue_ready(item_name, payload)

    worker = DeliveryWorker(
        spool=spool,
        client=_FakeClient(),
        batch_size=1,
    )
    worker._process_item(item_name=item_name + ".json", payload=payload)

    assert not (spool.ready_dir / f"{item_name}.json").exists()
