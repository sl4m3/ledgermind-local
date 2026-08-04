from __future__ import annotations

from typing import Any

from ledgermind_local.config import LocalConfig
from ledgermind_local.paths import ServicePaths

from .isolation import IsolationCapabilities, IsolationPlan, IsolationRequirements
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
        # bwrap uses this only for a Python-based Core test/runtime shim.
        "PYTHONHOME",
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
_REDACTED_ENVIRONMENT_MARKER = "<redacted>"


def _doctor_isolation_requirements(config: LocalConfig) -> IsolationRequirements:
    """Map the existing config switch to the independent Core guarantees."""

    protected = config.require_core_network_isolation
    return IsolationRequirements(
        require_network_isolation=protected,
        require_rounds_database_hidden=protected,
        require_filesystem_allowlist=protected,
        require_environment_sanitized=True,
        require_signature=config.verify_core_signature,
    )


def _sandbox_report(
    requirements: IsolationRequirements,
    capabilities: IsolationCapabilities,
) -> dict[str, Any]:
    """Build a compatibility-aware, capability-first sandbox report."""

    level = IsolationPlan(command=(), capabilities=capabilities).level
    return {
        "required": requirements.require_network_isolation,
        "level": level,
        "detail": capabilities.detail,
        "capabilities": capabilities.as_dict(),
        "requirements": requirements.as_dict(),
        "missing_requirements": list(requirements.missing(capabilities)),
    }


def _environment_report(
    environment_keys: list[str],
    capabilities: IsolationCapabilities,
) -> dict[str, Any]:
    """Return environment diagnostics without returning any environment names."""

    secret_like_count = sum(
        any(marker in key.upper() for marker in _SECRET_ENVIRONMENT_MARKERS)
        for key in environment_keys
    )
    unexpected_count = sum(
        key not in _SAFE_CORE_ENVIRONMENT_KEYS for key in environment_keys
    )
    return {
        # These lists are retained for the CLI compatibility contract, but
        # intentionally contain no Core environment names.
        "keys": [],
        "unexpected_keys": (
            [_REDACTED_ENVIRONMENT_MARKER] if unexpected_count else []
        ),
        "secret_like_keys": (
            [_REDACTED_ENVIRONMENT_MARKER] if secret_like_count else []
        ),
        "key_count": len(environment_keys),
        "unexpected_count": unexpected_count,
        "secret_like_count": secret_like_count,
        "sanitized": capabilities.environment_sanitized,
        "values_exposed": False,
    }


def build_core_doctor_report(
    *,
    paths: ServicePaths,
    config: LocalConfig,
) -> dict[str, Any]:
    """Collect secret-safe Core binary, sandbox, IPC and schema diagnostics."""

    binary = paths.resolve_core_path(config.core_binary_path)
    signature = paths.resolve_core_path(config.core_signature_path)
    public_key = paths.resolve_core_path(config.core_public_key_path)
    isolation_requirements = _doctor_isolation_requirements(config)
    unavailable_capabilities = IsolationCapabilities(
        sandbox_backend="unavailable",
        detail="Core was not launched",
    )
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
        "sandbox": _sandbox_report(isolation_requirements, unavailable_capabilities),
        "health": {
            "healthy": False,
            "detail": "Core was not launched",
            "protocol_version": None,
            "schema_version": None,
        },
        "environment": _environment_report([], unavailable_capabilities),
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
    signature_verified = signature_report["status"] == "verified"
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
            rounds_database_path=paths.resolve_rounds_database_path(
                config.rounds_database_path
            ),
            isolation_requirements=isolation_requirements,
            strict_isolation=config.require_core_network_isolation,
            binary_signature_verified=signature_verified,
        )
        try:
            health = supervisor.request("health", {})
            environment_keys = sorted(
                str(value) for value in health.get("environment_keys", [])
            )
            report["health"] = {
                "healthy": bool(health.get("healthy", False)),
                "detail": health.get("detail"),
                "protocol_version": health.get("protocol_version"),
                "schema_version": health.get("schema_version"),
            }
            handshake = supervisor.handshake_result or {}
            report["version"] = handshake.get("version")
            report["server_name"] = handshake.get("server_name")
            report["environment"] = _environment_report(
                environment_keys,
                supervisor.isolation_capabilities,
            )
        except CoreSupervisorError as exc:
            report["health"]["detail"] = str(exc)
        finally:
            report["sandbox"] = _sandbox_report(
                isolation_requirements,
                supervisor.isolation_capabilities,
            )
            supervisor.close()

    report["ok"] = bool(
        binary.is_file()
        and signature_allows_launch
        and report["health"]["healthy"]
        and report["health"]["protocol_version"] == 1
        and report["health"]["schema_version"] is not None
        and report["environment"]["unexpected_count"] == 0
        and report["environment"]["secret_like_count"] == 0
        and report["environment"]["sanitized"]
        and (
            not config.require_core_network_isolation
            or not report["sandbox"]["missing_requirements"]
        )
    )
    return report


__all__ = ["build_core_doctor_report"]