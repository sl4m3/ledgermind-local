"""Small deterministic Core IPC v1 process used by Local acceptance tests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _configure_python_paths() -> None:
    paths = [
        sys.argv[index + 1]
        for index, argument in enumerate(sys.argv[:-1])
        if argument == "--python-path" and index + 1 < len(sys.argv)
    ]
    for path in reversed(paths):
        sys.path.insert(0, path)


_configure_python_paths()

from ledgermind_protocol.core_ipc import (
    CORE_IPC_CAPABILITIES,
    CORE_IPC_OPERATIONS,
    CoreError,
    CoreRequestEnvelope,
    CoreResponseEnvelope,
)


def _read_frame() -> bytes:
    header = sys.stdin.buffer.read(4)
    if len(header) != 4:
        raise EOFError
    length = int.from_bytes(header, "big")
    payload = sys.stdin.buffer.read(length)
    if len(payload) != length:
        raise EOFError
    return payload


def _write(response: CoreResponseEnvelope) -> None:
    payload = response.to_json().encode("utf-8")
    sys.stdout.buffer.write(len(payload).to_bytes(4, "big") + payload)
    sys.stdout.buffer.flush()


def _write_raw(payload: bytes) -> None:
    sys.stdout.buffer.write(len(payload).to_bytes(4, "big") + payload)
    sys.stdout.buffer.flush()


def _argument_values(name: str) -> set[str]:
    values: set[str] = set()
    for index, argument in enumerate(sys.argv[:-1]):
        if argument == name:
            values.add(sys.argv[index + 1])
    return values


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _error(request_id: str, code: str, message: str) -> None:
    _write(
        CoreResponseEnvelope.from_error(
            request_id,
            CoreError(
                code=code,
                message=message,
                error_id="fake-error-1",
                retryable=False,
            ),
        )
    )


def _exchange_file(core_data_dir: Path, relative_path: str) -> Path:
    candidate = (core_data_dir / relative_path).resolve()
    root = core_data_dir.resolve()
    candidate.relative_to(root)
    return candidate


def main() -> int:
    crash_once_file: Path | None = None
    delay_seconds = 0.0
    stderr_line: str | None = None
    mismatched_health_id = "--mismatched-health-id" in sys.argv
    malformed_health_response = "--malformed-health-response" in sys.argv
    missing_capabilities = _argument_values("--missing-capability")
    missing_operations = _argument_values("--missing-operation")
    core_data_dir = Path(os.environ.get("LEDGERMIND_CORE_DATA_DIR", "."))
    idempotency: dict[str, str] = {}
    if "--crash-once-file" in sys.argv:
        marker_index = sys.argv.index("--crash-once-file") + 1
        crash_once_file = Path(sys.argv[marker_index])
    if "--delay-seconds" in sys.argv:
        delay_index = sys.argv.index("--delay-seconds") + 1
        delay_seconds = float(sys.argv[delay_index])
    if "--stderr-line" in sys.argv:
        stderr_index = sys.argv.index("--stderr-line") + 1
        stderr_line = sys.argv[stderr_index]
    if stderr_line is not None:
        print(stderr_line, file=sys.stderr, flush=True)

    while True:
        try:
            raw = _read_frame()
        except EOFError:
            return 0
        request = CoreRequestEnvelope.from_json(raw.decode("utf-8"))
        if request.operation == "handshake":
            operations = sorted(set(CORE_IPC_OPERATIONS) - missing_operations)
            capabilities = {
                capability: capability not in missing_capabilities
                for capability in CORE_IPC_CAPABILITIES
            }
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "protocol_version": 1,
                        "core_version": "fake-core-1",
                        "knowledge_schema_version": 6,
                        "supported_operations": operations,
                        "capabilities": capabilities,
                    },
                )
            )
        elif request.operation in missing_operations:
            _error(request.request_id, "INVALID_REQUEST", "operation is unavailable")
        elif request.operation == "health":
            if delay_seconds:
                time.sleep(delay_seconds)
            if crash_once_file is not None and not crash_once_file.exists():
                crash_once_file.parent.mkdir(parents=True, exist_ok=True)
                crash_once_file.touch()
                return 17
            if malformed_health_response:
                _write_raw(b"{not-json")
                return 18
            response_request_id = (
                f"{request.request_id}-wrong"
                if mismatched_health_id
                else request.request_id
            )
            _write(CoreResponseEnvelope.ok(response_request_id, {"healthy": True}))
        elif request.operation == "shutdown":
            _write(CoreResponseEnvelope.ok(request.request_id, {"stopped": True}))
            return 0
        elif request.operation == "accept_hypothesis":
            key = str(request.payload["idempotency_key"])
            fingerprint = json.dumps(
                request.payload, sort_keys=True, separators=(",", ":")
            )
            previous = idempotency.get(key)
            if previous is not None and previous != fingerprint:
                _write(
                    CoreResponseEnvelope.from_error(
                        request.request_id,
                        CoreError(
                            code="IDEMPOTENCY_CONFLICT",
                            message="idempotency key was already used with another payload",
                            error_id="fake-error-1",
                            retryable=False,
                        ),
                    )
                )
            else:
                duplicate = previous is not None
                idempotency[key] = fingerprint
                _write(
                    CoreResponseEnvelope.ok(
                        request.request_id,
                        {
                            "accepted": True,
                            "duplicate": duplicate,
                            "core_reference_id": "core-hypothesis-1",
                        },
                    )
                )
        elif request.operation == "retrieve_context":
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "api_version": "1",
                        "items": [
                            {
                                "knowledge_id": "knowledge-1",
                                "title": "Title",
                                "target": "Target",
                                "statement": "Statement",
                                "relevance": 0.9,
                            }
                        ],
                    },
                )
            )
        elif request.operation == "record_context_usage":
            _write(CoreResponseEnvelope.ok(request.request_id, {"recorded": True}))
        elif request.operation == "fail_model_task":
            payload = request.payload
            retryable = bool(payload.get("retryable", False))
            failed_at = str(payload.get("failed_at"))
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "status": "pending" if retryable else "failed",
                        "attempts": 1,
                        "available_at": failed_at if retryable else None,
                        "last_error_code": str(payload.get("error_code", "unknown")),
                        "failed_at": failed_at,
                        "completed_at": None if retryable else failed_at,
                    },
                )
            )
        elif request.operation == "create_backup":
            relative_path = "exchange/outgoing/fake-core-backup.bin"
            snapshot = b"fake-core-snapshot-v1"
            destination = _exchange_file(core_data_dir, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(snapshot)
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "relative_path": relative_path,
                        "sha256": _digest(snapshot),
                        "size_bytes": len(snapshot),
                        "schema_version": 6,
                    },
                )
            )
        elif request.operation in {"validate_backup", "prepare_restore", "begin_restore"}:
            raw_relative_path = request.payload.get("relative_path")
            expected_sha = request.payload.get("sha256")
            if (
                not isinstance(raw_relative_path, str)
                or not raw_relative_path.startswith("exchange/incoming/")
                or ".." in raw_relative_path.split("/")
                or not isinstance(expected_sha, str)
            ):
                _error(request.request_id, "INVALID_REQUEST", "invalid exchange path")
                continue
            relative_path = raw_relative_path
            try:
                source = _exchange_file(core_data_dir, relative_path)
                content = source.read_bytes()
            except (OSError, ValueError):
                _error(request.request_id, "INTEGRITY_VIOLATION", "backup artifact is missing")
                continue
            actual_sha = _digest(content)
            if actual_sha != expected_sha:
                _error(request.request_id, "INTEGRITY_VIOLATION", "backup digest mismatch")
                continue
            result: dict[str, object] = {
                "relative_path": relative_path,
                "sha256": actual_sha,
                "size_bytes": len(content),
                "schema_version": 6,
            }
            if request.operation == "begin_restore":
                if request.payload.get("restore_token") != "fake-restore-token-1":
                    _error(request.request_id, "INVALID_REQUEST", "restore token is invalid")
                    continue
                result.update(
                    {
                        "restore_transaction_id": "fake-restore-transaction-1",
                        "state": "applied_pending_commit",
                    }
                )
            elif request.operation == "prepare_restore":
                result.update(
                    {
                        "restore_token": "fake-restore-token-1",
                        "requires_restart": True,
                    }
                )
            _write(CoreResponseEnvelope.ok(request.request_id, result))
        elif request.operation == "commit_restore":
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "restore_transaction_id": request.payload.get("restore_transaction_id"),
                        "committed": True,
                        "state": "committed",
                    },
                )
            )
        elif request.operation == "rollback_restore":
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "restore_transaction_id": request.payload.get("restore_transaction_id"),
                        "rolled_back": True,
                        "state": "rolled_back",
                    },
                )
            )
        else:
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {"accepted": True, "duplicate": False},
                )
            )


if __name__ == "__main__":
    raise SystemExit(main())
