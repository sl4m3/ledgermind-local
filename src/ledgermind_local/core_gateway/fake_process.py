"""Small deterministic Core IPC v1 process used by Local acceptance tests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ledgermind_protocol.core_ipc import (
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


def main() -> int:
    crash_once_file: Path | None = None
    delay_seconds = 0.0
    stderr_line: str | None = None
    mismatched_health_id = "--mismatched-health-id" in sys.argv
    malformed_health_response = "--malformed-health-response" in sys.argv
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
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {
                        "protocol_version": 1,
                        "server_name": "fake-core",
                        "operations": [
                            "handshake",
                            "health",
                            "accept_hypothesis",
                            "retrieve_context",
                            "record_context_usage",
                            "shutdown",
                        ],
                    },
                )
            )
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
        else:
            _write(
                CoreResponseEnvelope.ok(
                    request.request_id,
                    {"accepted": True, "duplicate": False},
                )
            )


if __name__ == "__main__":
    raise SystemExit(main())
