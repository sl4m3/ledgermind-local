"""Filesystem spool for plugin delivery tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_STATE_FILE = "state.json"


@dataclass(frozen=True, slots=True)
class SpoolStats:
    pending_count: int
    ready_count: int
    failed_count: int


class SpoolStateError(RuntimeError):
    """State file in plugin spool is invalid or not parseable."""


class FileSpool:
    """Minimal file spool with atomic writes and a tiny per-spool state file."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending-extraction"
        self.ready_dir = self.root / "ready-delivery"
        self.failed_dir = self.root / "failed"
        self.state_file = self.root / _STATE_FILE

    @property
    def state_path(self) -> Path:
        return self.state_file

    def _ensure(self) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {
                "version": 1,
                "last_completed_message_id_by_session": {},
            }
        payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SpoolStateError("invalid spool state")
        checkpoints = payload.get("last_completed_message_id_by_session", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
        normalized = {str(k): str(v) for k, v in checkpoints.items()}
        return {
            "version": int(payload.get("version", 1)) or 1,
            "last_completed_message_id_by_session": normalized,
        }

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self._ensure()
        destination = self.state_file
        with destination.with_suffix(".tmp").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        destination.with_suffix(".tmp").replace(destination)

    def get_last_completed_message_id(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        try:
            return self._read_state()["last_completed_message_id_by_session"].get(session_id)  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError, SpoolStateError):
            return None

    def set_last_completed_message_id(self, session_id: str, message_id: str | None) -> None:
        if not session_id or message_id is None:
            return
        try:
            state = self._read_state()
            checkpoints = dict(state.get("last_completed_message_id_by_session", {}))
            existing = checkpoints.get(session_id)
            if existing == message_id:
                return
            checkpoints[session_id] = str(message_id)
            state["last_completed_message_id_by_session"] = checkpoints
            self._write_state(state)
        except (OSError, json.JSONDecodeError, SpoolStateError):
            self._write_state(
                {
                    "version": 1,
                    "last_completed_message_id_by_session": {session_id: str(message_id)},
                }
            )

    def _write_atomic(self, destination: Path, payload: Mapping[str, Any]) -> Path:
        self._ensure()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        return destination

    def _normalize_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a mapping")
        return json.loads(json.dumps(payload, ensure_ascii=False))  # type: ignore[no-any-return]

    def _is_same_payload(self, path: Path, payload: Mapping[str, Any]) -> bool:
        if not path.exists():
            return False
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            return existing == payload  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            return False

    def enqueue_pending(self, idempotency_key: str, payload: Mapping[str, Any]) -> Path:
        path = self.pending_dir / f"{idempotency_key}.json"
        normalized = self._normalize_payload(payload)
        if self._is_same_payload(path, normalized):
            return path
        return self._write_atomic(path, normalized)

    def enqueue_ready(self, idempotency_key: str, payload: Mapping[str, Any]) -> Path:
        path = self.ready_dir / f"{idempotency_key}.json"
        normalized = self._normalize_payload(payload)
        if self._is_same_payload(path, normalized):
            return path
        return self._write_atomic(path, normalized)

    def replace_ready(self, item_name: str, payload: Mapping[str, Any]) -> Path:
        path = self.ready_dir / item_name
        return self._write_atomic(path, self._normalize_payload(payload))

    def pop_ready(self, limit: int = 1) -> list[tuple[str, Mapping[str, Any]]]:
        self._ensure()
        items: list[tuple[str, Mapping[str, Any]]] = []
        for file_path in sorted(self.ready_dir.glob("*.json"))[:limit]:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                file_path.unlink(missing_ok=True)
                continue
            if not isinstance(payload, dict):
                self.fail_item(file_path.name)
                continue
            items.append((file_path.name, payload))
        return items

    def complete_item(self, item_name: str) -> None:
        target = self.ready_dir / item_name
        target.unlink(missing_ok=True)

    def fail_item(self, item_name: str) -> Path:
        source = self.ready_dir / item_name
        target = self.failed_dir / item_name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        return target

    def mark_failed(self, item_name: str, reason: str | None = None) -> Path:
        source = self.ready_dir / item_name
        if not source.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.failed_dir / item_name
            return target

        payload = {}
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
        except (OSError, json.JSONDecodeError):
            payload = {}
        if reason is not None:
            delivery = dict(payload.get("delivery", {}))
            delivery["failure_reason"] = str(reason)
            payload["delivery"] = delivery

        self.failed_dir.mkdir(parents=True, exist_ok=True)
        target = self._write_atomic(self.failed_dir / item_name, payload)
        source.unlink(missing_ok=True)
        return target

    def mark_retry(self, item_name: str, payload: Mapping[str, Any]) -> None:
        self.replace_ready(item_name, payload)

    def stats(self) -> SpoolStats:
        return SpoolStats(
            pending_count=len(list(self.pending_dir.glob("*.json"))),
            ready_count=len(list(self.ready_dir.glob("*.json"))),
            failed_count=len(list(self.failed_dir.glob("*.json"))),
        )
