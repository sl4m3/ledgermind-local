from __future__ import annotations

from typing import Any

from ledgermind_local.config import LocalConfig
from ledgermind_local.paths import ServicePaths

from .sandbox import SandboxLevel
from .signing import (
    CoreBinaryVerificationError,
    calculate_sha256,
    verify_core_binary,
)
from .supervisor import CoreSupervisor, CoreSupervisorError

_SAFE_CORE_ENVIRONMENT_KEYS = frozenset(
    {
        "LANG",
        "LC_ALL",
        "RUST_BACKTRACE",
        "LEDGERMIND_CORE_DATA_DIR",
        # bwrap derives this from the requested working directory.
        "PWD",
    }
)
_SECRET_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PROXY",
    "SECRET",
    "TOKEN",
)


def build_core_doctor_report(
    *,
    paths: ServicePaths,
    config: LocalConfig,
) -> dict[str, Any]:
    """Collect secret-safe Core binary, sandbox, IPC and schema diagnostics."""

    binary = paths.resolve_core_path(config.core_binary_path)
    signature = paths.resolve_core_path(config.core_signature_path)
    public_key = paths.resolve_core_path(config.core_public_key_path)
    report: dict[str, Any] = {
        "ok": False,
        "backend": config.core_backend,
        "version": None,
        "server_name": None,
        "binary": {
            "path": str(binary),
            "present": binary.is_file(),
            "sha256": None,
        },
        "signature": {
            "required": config.verify_core_signature,
            "path": str(signature),
            "public_key_path": str(public_key),
            "status": "not_checked",
            "detail": None,
        },
        "sandbox": {
            "required": config.require_core_network_isolation,
            "level": SandboxLevel.UNAVAILABLE.value,
            "detail": "Core was not launched",
        },
        "health": {
            "healthy": False,
            "detail": "Core was not launched",
            "protocol_version": None,
            "schema_version": None,
        },
        "environment": {
            "keys": [],
            "unexpected_keys": [],
            "secret_like_keys": [],
            "values_exposed": False,
        },
    }

    if binary.is_file():
        try:
            report["binary"]["sha256"] = calculate_sha256(binary)
        except OSError as exc:
            report["binary"]["detail"] = f"cannot hash Core binary: {exc}"

    signature_report = report["signature"]
    if not binary.is_file():
        signature_report["status"] = "failed"
        signature_report["detail"] = f"Core binary not found: {binary}"
    elif not config.verify_core_signature:
        signature_report["status"] = "not_required"
    else:
        try:
            verification = verify_core_binary(
                binary,
                signature_path=signature,
                public_key_path=public_key,
            )
            report["binary"]["sha256"] = verification.sha256
            signature_report["status"] = "verified"
        except CoreBinaryVerificationError as exc:
            signature_report["status"] = "failed"
            signature_report["detail"] = str(exc)

    signature_allows_launch = signature_report["status"] in {
        "verified",
        "not_required",
    }
    if binary.is_file() and signature_allows_launch:
        supervisor = CoreSupervisor(
            [
                str(binary),
                "--database",
                str(paths.resolve_knowledge_database_path(config.knowledge_database_path)),
            ],
            startup_timeout_seconds=config.core_startup_timeout_seconds,
            operation_timeout_seconds=config.core_request_timeout_seconds,
            core_data_dir=paths.core_data_dir,
            blocked_data_dirs=(paths.home,),
            require_network_isolation=config.require_core_network_isolation,
        )
        try:
            health = supervisor.request("health", {})
            environment_keys = sorted(
                str(value) for value in health.get("environment_keys", [])
            )
            secret_like_keys = [
                key
                for key in environment_keys
                if any(marker in key.upper() for marker in _SECRET_ENVIRONMENT_MARKERS)
            ]
            unexpected_keys = [
                key for key in environment_keys if key not in _SAFE_CORE_ENVIRONMENT_KEYS
            ]
            report["health"] = {
                "healthy": bool(health.get("healthy", False)),
                "detail": health.get("detail"),
                "protocol_version": health.get("protocol_version"),
                "schema_version": health.get("schema_version"),
            }
            handshake = supervisor.handshake_result or {}
            report["version"] = handshake.get("version")
            report["server_name"] = handshake.get("server_name")
            report["environment"] = {
                "keys": environment_keys,
                "unexpected_keys": unexpected_keys,
                "secret_like_keys": secret_like_keys,
                "values_exposed": False,
            }
        except CoreSupervisorError as exc:
            report["health"]["detail"] = str(exc)
        finally:
            report["sandbox"] = {
                "required": config.require_core_network_isolation,
                "level": supervisor.sandbox_level,
                "detail": supervisor.sandbox_detail,
            }
            supervisor.close()

    report["ok"] = bool(
        binary.is_file()
        and signature_allows_launch
        and report["health"]["healthy"]
        and report["health"]["protocol_version"] == 1
        and report["health"]["schema_version"] is not None
        and not report["environment"]["unexpected_keys"]
        and not report["environment"]["secret_like_keys"]
        and (
            not config.require_core_network_isolation
            or report["sandbox"]["level"] == SandboxLevel.FULL.value
        )
    )
    return report


__all__ = ["build_core_doctor_report"]