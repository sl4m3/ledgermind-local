"""Lifecycle and serialized request transport for a Core child process."""

from __future__ import annotations

import os
import select
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ledgermind_protocol.core_ipc import (
    CORE_IPC_OPERATIONS,
    CoreError,
    CoreRequestEnvelope,
    CoreResponseEnvelope,
)

from .framing import FrameError, read_frame, write_frame
from .sandbox import SandboxLevel, SandboxUnavailableError, build_sandbox_plan


class CoreSupervisorError(RuntimeError):
    """Base error for Core process lifecycle and transport failures."""


class CoreSupervisorTimeout(CoreSupervisorError):
    """The Core process did not answer before the operation deadline."""


class CoreSupervisorCrashed(CoreSupervisorError):
    """The Core process exited or its protocol stream broke."""


class CoreSupervisorRemoteError(CoreSupervisorError):
    """Core returned a structured domain/protocol error response."""

    def __init__(self, error: CoreError) -> None:
        self.code = error.code
        self.detail = error.message
        self.retryable = error.retryable
        super().__init__(f"Core returned {self.code}: {self.detail}")


class CoreSupervisor:
    """Own one long-lived Core process and serialize its request/response stream."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_seconds: float = 5.0,
        operation_timeout_seconds: float = 5.0,
        client_name: str = "ledgermind-local",
        client_version: str = "4.0.0a1",
        core_data_dir: str | Path | None = None,
        blocked_data_dirs: Sequence[str | Path] = (),
        require_network_isolation: bool = False,
    ) -> None:
        if not command or any(not str(part) for part in command):
            raise ValueError("Core command must not be empty")
        self._command = tuple(str(part) for part in command)
        self._startup_timeout = max(float(startup_timeout_seconds), 0.01)
        self._operation_timeout = max(float(operation_timeout_seconds), 0.01)
        self._client_name = client_name
        self._client_version = client_version
        self._core_data_dir = Path(core_data_dir or Path.cwd()).expanduser()
        self._blocked_data_dirs = tuple(
            Path(item).expanduser() for item in blocked_data_dirs
        )
        self._require_network_isolation = require_network_isolation
        self._sandbox_level = SandboxLevel.UNAVAILABLE
        self._sandbox_detail = "Core has not been launched"
        self._handshake_result: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._is_running_locked():
                return
            self._spawn_locked()
            try:
                result = self._exchange_locked(
                    "handshake",
                    {
                        "client_name": self._client_name,
                        "client_version": self._client_version,
                        "supported_operations": sorted(CORE_IPC_OPERATIONS),
                    },
                    timeout=self._startup_timeout,
                )
                if int(result.get("protocol_version", 0)) != 1:
                    raise CoreSupervisorError(
                        "Core handshake returned unsupported version"
                    )
                self._handshake_result = result
            except Exception:
                self._terminate_locked()
                raise

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.start()
            try:
                return self._exchange_locked(
                    operation,
                    dict(payload or {}),
                    request_id=request_id,
                    timeout=timeout_seconds or self._operation_timeout,
                )
            except CoreSupervisorTimeout:
                self._terminate_locked()
                raise
            except CoreSupervisorRemoteError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                self._terminate_locked()
                raise CoreSupervisorError("invalid Core response") from exc
            except CoreSupervisorError:
                self._terminate_locked()
                raise
            except (BrokenPipeError, ConnectionError, FrameError, OSError) as exc:
                self._terminate_locked()
                raise CoreSupervisorCrashed("Core protocol stream failed") from exc

    def close(self) -> None:
        with self._lock:
            if self._is_running_locked():
                try:
                    self._exchange_locked(
                        "shutdown", {}, timeout=self._operation_timeout
                    )
                except (CoreSupervisorError, FrameError, OSError, ValueError):
                    self._terminate_locked()
            self._terminate_locked()

    def restart(self) -> None:
        with self._lock:
            self._terminate_locked()
            self.start()

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._process.pid if self._process is not None else None

    def stderr_lines(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._stderr_lines)

    @property
    def sandbox_level(self) -> str:
        with self._lock:
            return self._sandbox_level.value

    @property
    def sandbox_detail(self) -> str:
        with self._lock:
            return self._sandbox_detail

    @property
    def handshake_result(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._handshake_result) if self._handshake_result else None

    def _spawn_locked(self) -> None:
        try:
            sandbox_plan = build_sandbox_plan(
                self._command,
                core_data_dir=self._core_data_dir,
                blocked_data_dirs=self._blocked_data_dirs,
                required=self._require_network_isolation,
            )
        except SandboxUnavailableError as exc:
            self._sandbox_level = SandboxLevel.UNAVAILABLE
            self._sandbox_detail = str(exc)
            raise CoreSupervisorError(str(exc)) from exc

        self._sandbox_level = sandbox_plan.level
        self._sandbox_detail = sandbox_plan.detail
        try:
            self._process = subprocess.Popen(
                sandbox_plan.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                cwd=str(self._core_data_dir),
                env=self._build_environment(),
            )
        except OSError as exc:
            raise CoreSupervisorCrashed("failed to start Core process") from exc
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            name="ledgermind-core-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _build_environment(self) -> dict[str, str]:
        environment = {
            "RUST_BACKTRACE": "0",
            "LEDGERMIND_CORE_DATA_DIR": str(self._core_data_dir),
        }
        for name in ("LANG", "LC_ALL"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment

    def _exchange_locked(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise CoreSupervisorCrashed("Core process is not running")
        envelope = CoreRequestEnvelope(
            protocol_version=1,
            request_id=request_id or str(uuid.uuid4()),
            operation=operation,
            payload=payload,
        )
        encoded = envelope.to_json().encode("utf-8")
        write_frame(process.stdin, encoded)
        response = CoreResponseEnvelope.from_json(
            self._read_frame_with_timeout(process.stdout, timeout).decode("utf-8")
        )
        if response.request_id != envelope.request_id:
            raise CoreSupervisorError("Core response request_id does not match request")
        if response.status == "error":
            assert response.error is not None
            raise CoreSupervisorRemoteError(response.error)
        return dict(response.result or {})

    @staticmethod
    def _read_frame_with_timeout(stream: Any, timeout: float) -> bytes:
        deadline = time.monotonic() + max(timeout, 0.01)
        if not hasattr(stream, "fileno"):
            return read_frame(stream)
        chunks = bytearray()
        header = CoreSupervisor._read_exact_with_timeout(stream, 4, deadline)
        length = int.from_bytes(header, "big")
        if length > 8 * 1024 * 1024:
            raise FrameError("Core frame exceeds maximum size")
        chunks.extend(CoreSupervisor._read_exact_with_timeout(stream, length, deadline))
        return bytes(chunks)

    @staticmethod
    def _read_exact_with_timeout(stream: Any, size: int, deadline: float) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise CoreSupervisorTimeout("Core response timed out")
            ready, _, _ = select.select([stream], [], [], wait)
            if not ready:
                raise CoreSupervisorTimeout("Core response timed out")
            chunk = stream.read(remaining)
            if not chunk:
                raise CoreSupervisorCrashed("Core process closed stdout")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def _drain_stderr(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    with self._lock:
                        self._stderr_lines.append(line)
        except (OSError, ValueError):
            return


__all__ = [
    "CoreSupervisor",
    "CoreSupervisorCrashed",
    "CoreSupervisorError",
    "CoreSupervisorRemoteError",
    "CoreSupervisorTimeout",
]
