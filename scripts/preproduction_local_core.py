"""Real Local -> Core -> provider pre-production gate.

This command deliberately launches the installed Local CLI and lets Local
launch the signed Rust Core through its normal supervisor.  It does not use a
fake Core, a fake provider, or a direct Lab-to-Core shortcut.  The only
fixtures are the captured Hermes Round A and the repository's Round B YAML.

The command is opt-in because it makes real provider requests::

    PYTHONPATH=src:../ledgermind-integrations/protocol/python/src \
      python scripts/preproduction_local_core.py \
        --lab-config ~/.config/ledgermind-lab/config.json \
        --core-binary ../ledgermind-core/target/release/ledgermind-core

All output is secret-free.  Provider credentials are copied only into a
0600 temporary Local secret store and are never included in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_LOCAL_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _LOCAL_ROOT.parent
_LAB_ROOT = _WORKSPACE_ROOT / "ledgermind-lab"
_INTEGRATIONS_PROTOCOL_ROOT = _WORKSPACE_ROOT / "ledgermind-integrations" / "protocol" / "python"
for _path in (
    _LOCAL_ROOT / "src",
    _LAB_ROOT / "src",
    _INTEGRATIONS_PROTOCOL_ROOT / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ledgermind_lab.config import LabConfig, load_config  # noqa: E402
from ledgermind_lab.scenarios.compiler import compile_scenario  # noqa: E402
from ledgermind_lab.scenarios.loader import load_scenario_file  # noqa: E402
from ledgermind_local.config import (  # noqa: E402
    CoreSecurityConfig,
    EmbeddingConfig,
    LocalConfig,
    WorkerConfig,
    WorkerSetConfig,
)
from ledgermind_local.inference.profile_store import InferenceProfileStore  # noqa: E402
from ledgermind_local.inference.profiles import InferenceProfile  # noqa: E402
from ledgermind_local.inference.provider_probe import ProviderProbe  # noqa: E402
from ledgermind_local.inference.profile_slots import DatabaseBackedProfileResolver, ProfileSlot  # noqa: E402
from ledgermind_local.inference.secrets import SecretStore  # noqa: E402
from ledgermind_local.paths import ServicePaths  # noqa: E402
from ledgermind_local.persistence import open_sqlite_connection  # noqa: E402
from ledgermind_local.persistence import rounds_migrations  # noqa: E402
from ledgermind_local.inference.profile_store import DatabaseBackedCapabilityStore  # noqa: E402


class GateFailure(RuntimeError):
    """A hard pre-production gate failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _round_payloads(lab_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = lab_root / "scenarios" / "suites" / "operational" / "hermes-locale-market-real"
    # The capture keeps its historical digest as immutable evidence.  The
    # normal Lab compiler overlays replay identity and emits the canonical
    # digest required by the production Local API.
    round_a = compile_scenario(load_scenario_file(suite / "scenario.yaml")).raw_round.model_dump(
        mode="json", exclude_none=True
    )
    round_b = compile_scenario(load_scenario_file(suite / "round-b.yaml")).raw_round
    return round_a, round_b.model_dump(mode="json", exclude_none=True)


def _copy_signed_core(binary: Path, paths: ServicePaths) -> None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise GateFailure(f"real Core binary is missing or not executable: {binary}")
    target = paths.core_data_dir / "bin" / "ledgermind-core"
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy2(binary, target)
    os.chmod(target, 0o700)
    private_key = Ed25519PrivateKey.generate()
    (target.with_suffix(".sig")).write_bytes(private_key.sign(target.read_bytes()))
    (target.with_suffix(".pub")).write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )


def _config_for(
    *,
    home: Path,
    lab_config: LabConfig,
    port: int,
) -> LocalConfig:
    generation = lab_config.generation
    embedding = lab_config.embedding
    return LocalConfig(
        config_version=2,
        bind_host="127.0.0.1",
        bind_port=port,
        rounds_database_path="rounds.db",
        knowledge_database_path="knowledge.db",
        core_backend="process",
        core_binary_path="bin/ledgermind-core",
        core_signature_path="bin/ledgermind-core.sig",
        core_public_key_path="bin/ledgermind-core.pub",
        verify_core_signature=True,
        core_startup_timeout_seconds=30.0,
        core_request_timeout_seconds=60.0,
        core_security=CoreSecurityConfig(
            profile="secure",
            require_network_isolation=True,
            require_rounds_database_hidden=True,
            require_filesystem_allowlist=True,
            require_environment_sanitized=True,
            require_signature=True,
        ),
        workers=WorkerSetConfig(
            retention=WorkerConfig(enabled=False),
            core_commands=WorkerConfig(enabled=True, interval_seconds=0.2),
            core_model_tasks=WorkerConfig(enabled=True, interval_seconds=0.2),
        ),
        worker_max_attempts=3,
        worker_retry_delay_seconds=1.0,
        # The lease must exceed the provider timeout.  Reasoning-capable
        # models can spend most of that timeout before emitting JSON; a
        # shorter lease would duplicate the same Core task and break the
        # per-round generation budget.
        worker_lease_seconds=420.0,
        generation_concurrency=1,
        provider_capability_ttl_seconds=86_400,
        inference_secrets_path=str(home / "secrets.json"),
        embedding=EmbeddingConfig(
            enabled=True,
            provider_mode="api",
            endpoint=embedding.endpoint,
            model=embedding.model,
            dimensions=embedding.dimensions or 2048,
            batch_size=embedding.batch_size,
            max_concurrency=embedding.max_concurrency,
            max_wait_seconds=embedding.max_wait_seconds,
            timeout_seconds=min(embedding.timeout_seconds, 120.0),
            secret_ref="embedding-api",
        ),
    )


def _seed_local(
    *,
    paths: ServicePaths,
    config: LocalConfig,
    lab_config: LabConfig,
    memory_space_id: str,
) -> None:
    generation = lab_config.generation
    embedding = lab_config.embedding
    SecretStore(paths.home / "secrets.json").put(
        "generation-api", generation.token.get_secret_value()
    )
    SecretStore(paths.home / "secrets.json").put(
        "embedding-api", embedding.token.get_secret_value()
    )
    database = paths.resolve_rounds_database_path(config.rounds_database_path)
    connection = open_sqlite_connection(database)
    try:
        rounds_migrations.apply_migrations(connection)
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_space_id) DO NOTHING
            """,
            (memory_space_id, "hermes-production-gate", now, now),
        )
        store = InferenceProfileStore(connection)
        reasoning_profile = generation.reasoning.for_provider_model(
            generation.provider,
            generation.model,
        )
        generation_extra_body = dict(reasoning_profile.extra_body)
        if (
            reasoning_profile.reasoning_effort is not None
            and "reasoning_effort" not in generation_extra_body
        ):
            generation_extra_body["reasoning_effort"] = (
                reasoning_profile.reasoning_effort
            )
        for profile_id, slot in (
            ("generation-operational", "operational"),
            ("generation-background", "background"),
        ):
            store.upsert(
                InferenceProfile(
                    profile_id=profile_id,
                    base_url=generation.endpoint,
                    model=generation.model,
                    secret_ref="generation-api",
                    timeout_seconds=min(generation.timeout_seconds, 300.0),
                    max_retries=generation.max_retries,
                    max_input_tokens=12_000,
                    max_output_tokens=generation.max_output_tokens,
                    extra_body=generation_extra_body,
                    structured_output_preference="auto",
                    token_parameter="max_tokens",
                    supports_system_role=True,
                    supports_seed=False,
                )
            )
            store.bind_slot(memory_space_id, slot=slot, profile_id=profile_id)
        store.upsert(
            InferenceProfile(
                profile_id="embedding-api",
                base_url=embedding.endpoint,
                model=embedding.model,
                secret_ref="embedding-api",
                timeout_seconds=min(embedding.timeout_seconds, 120.0),
                max_retries=embedding.max_retries,
                max_input_tokens=12_000,
                max_output_tokens=2_000,
            )
        )
        store.bind_slot(memory_space_id, slot="embedding", profile_id="embedding-api")
        connection.commit()
    finally:
        connection.close()


def _probe_generation_profiles(*, paths: ServicePaths, database: Path, memory_space_id: str) -> dict[str, Any]:
    """Warm both generation capability-cache rows with real provider calls."""

    connection = open_sqlite_connection(database)
    try:
        resolver = DatabaseBackedProfileResolver(database)
        probe = ProviderProbe(
            profile_resolver=resolver,
            secret_store=SecretStore(paths.home / "secrets.json"),
            capability_store=DatabaseBackedCapabilityStore(str(database)),
            capability_ttl_seconds=86_400,
        )
        results: dict[str, Any] = {}
        for slot in (ProfileSlot.OPERATIONAL, ProfileSlot.BACKGROUND):
            result = probe.probe(memory_space_id, slot, force=False, reprobe=False)
            results[slot.value] = {
                "status": result.status,
                "selected_mode": result.selected_mode,
                "cache_hit": bool(result.metadata.get("cache_hit", False)),
                "profile_fingerprint": result.capabilities.profile_fingerprint,
                "probe_result": result.capabilities.probe_result,
                "error_code": result.error_code,
            }
        return results
    finally:
        connection.close()


def _http_json(
    base_url: str,
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    headers = {"accept": "application/json", "authorization": f"Bearer {token}"}
    if body is not None:
        headers["content-type"] = "application/json"
    request = urllib_request.Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=10.0) as response:
            status = response.status
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
    except urllib_error.URLError as exc:
        return 0, {"error": "local_unreachable", "detail": str(exc.reason)[:200]}
    try:
        decoded = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        decoded = {"error": "non_json_response"}
    return status, decoded if isinstance(decoded, dict) else {"value": decoded}


def _wait_until(
    predicate: Callable[[], bool], *, timeout: float, interval: float = 1.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    if not predicate():
        raise GateFailure("timeout waiting for production Local/Core queue")


def _local_counts(database: Path, memory_space_id: str) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        def count(table: str, where: str = "") -> int:
            suffix = f" WHERE {where}" if where else ""
            row = connection.execute(f"SELECT count(*) FROM {table}{suffix}", (memory_space_id,)).fetchone()
            return int(row[0]) if row else 0

        return {
            "raw_rounds": count("raw_rounds", "memory_space_id = ?"),
            "processing_jobs": count(
                "round_processing_jobs",
                "raw_round_id IN (SELECT raw_round_id FROM raw_rounds WHERE memory_space_id = ?)",
            ),
            "hypotheses": count(
                "hypotheses",
                "raw_round_id IN (SELECT raw_round_id FROM raw_rounds WHERE memory_space_id = ?)",
            ),
            "core_commands": count("core_commands", "memory_space_id = ?"),
        }
    finally:
        connection.close()


def _core_counts(database: Path, memory_space_id: str) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        params = (memory_space_id,)
        objects = connection.execute(
            "SELECT count(*) FROM memory_objects WHERE memory_space_id = ?", params
        ).fetchone()
        values = connection.execute(
            "SELECT count(*) FROM knowledge_values WHERE memory_space_id = ?", params
        ).fetchone()
        active_values = connection.execute(
            """
            SELECT count(*) FROM knowledge_values
            WHERE memory_space_id = ? AND superseded_by_value_id IS NULL
            """,
            params,
        ).fetchone()
        return {
            "objects": int(objects[0]) if objects else 0,
            "values": int(values[0]) if values else 0,
            "active_values": int(active_values[0]) if active_values else 0,
        }
    finally:
        connection.close()


def _core_task_state(database: Path) -> dict[str, Any]:
    """Read live Core task state, independent of Local's startup snapshot.

    Local captures object-facet statistics before starting its workers.  The
    resulting health snapshot may therefore retain the number of embedding
    tasks that were queued during startup even after the real Core worker has
    completed them.  The production gate must use the authoritative Core DB
    for the drain assertion while still recording Local health separately.
    """

    connection = sqlite3.connect(database)
    try:
        pending = connection.execute(
            """
            SELECT count(*) FROM model_tasks
            WHERE status IN ('queued', 'pending', 'leased')
            """
        ).fetchone()
        terminal_failures = connection.execute(
            """
            SELECT count(*) FROM model_tasks
            WHERE status = 'failed' OR last_error_code IS NOT NULL
            """
        ).fetchone()
        completed = connection.execute(
            "SELECT count(*) FROM model_tasks WHERE status = 'completed'"
        ).fetchone()
        return {
            "pending": int(pending[0]) if pending else 0,
            "terminal_failures": int(terminal_failures[0]) if terminal_failures else 0,
            "completed": int(completed[0]) if completed else 0,
        }
    finally:
        connection.close()


def _core_round_status(database: Path, raw_round_id: str | None) -> str | None:
    if not raw_round_id:
        return None
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT status FROM operational_round_states WHERE raw_round_id = ?",
            (raw_round_id,),
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        connection.close()


def _core_task_counts(database: Path, raw_round_id: str | None = None) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        if raw_round_id is None:
            rows = connection.execute(
                "SELECT task_type, count(*) FROM model_tasks GROUP BY task_type"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT task_type, count(*) FROM model_tasks
                WHERE json_extract(payload_json, '$.operation_input.raw_round_id') = ?
                GROUP BY task_type
                """,
                (raw_round_id,),
            ).fetchall()
        return {str(task_type): int(count) for task_type, count in rows}
    finally:
        connection.close()


def _core_generation_task_counts(database: Path, raw_round_id: str) -> dict[str, int]:
    counts = _core_task_counts(database, raw_round_id)
    return {
        task_type: counts.get(task_type, 0)
        for task_type in ("extract_claims", "resolve_subjects")
    }


def _core_embedding_counts(database: Path) -> dict[str, int]:
    """Return the persisted vector counts by Core entity type."""

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT entity_type, count(*)
            FROM entity_embeddings
            GROUP BY entity_type
            """
        ).fetchall()
        return {str(entity_type): int(count) for entity_type, count in rows}
    finally:
        connection.close()


def _local_delivery_state(database: Path, raw_round_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        command = connection.execute(
            """
            SELECT status, attempts, last_error_code
            FROM core_commands
            WHERE command_id = (
                SELECT command_id FROM raw_round_core_deliveries
                WHERE raw_round_id = ?
            )
            """,
            (raw_round_id,),
        ).fetchone()
        delivery = connection.execute(
            """
            SELECT transport_status, core_raw_round_id, last_error_code
            FROM raw_round_core_deliveries
            WHERE raw_round_id = ?
            """,
            (raw_round_id,),
        ).fetchone()
        return {
            "command_status": str(command["status"]) if command is not None else None,
            "command_attempts": int(command["attempts"]) if command is not None else 0,
            "command_error_code": (
                str(command["last_error_code"])
                if command is not None and command["last_error_code"] is not None
                else None
            ),
            "transport_status": (
                str(delivery["transport_status"]) if delivery is not None else None
            ),
            "core_raw_round_id": (
                str(delivery["core_raw_round_id"])
                if delivery is not None and delivery["core_raw_round_id"] is not None
                else None
            ),
            "transport_error_code": (
                str(delivery["last_error_code"])
                if delivery is not None and delivery["last_error_code"] is not None
                else None
            ),
        }
    finally:
        connection.close()


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _start_local(
    *,
    home: Path,
    port: int,
    env: Mapping[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "ledgermind_local.cli",
        "--home",
        str(home),
        "serve",
        "--port",
        str(port),
    ]
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=_LOCAL_ROOT,
        env=dict(env),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    # The descriptor is inherited by the child; closing the parent copy is
    # safe and makes cleanup deterministic.
    log_handle.close()
    return process


def run_gate(
    *,
    lab_config_path: Path,
    core_binary: Path,
    lab_root: Path,
    timeout: float,
    keep_work_dir: Path | None,
) -> dict[str, Any]:
    lab_config = load_config(lab_config_path)
    round_a, round_b = _round_payloads(lab_root)
    memory_space_id = str(round_a["memory_space_id"])
    work_context = tempfile.TemporaryDirectory(prefix="ledgermind-preprod-")
    work_dir = Path(work_context.name)
    if keep_work_dir is not None:
        work_context.cleanup()
        work_dir = keep_work_dir
        work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        work_context = None  # type: ignore[assignment]
    home = work_dir / "local"
    paths = ServicePaths(home)
    port = _free_port()
    config = _config_for(home=home, lab_config=lab_config, port=port)
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.config_file.write_text(config.to_json(), encoding="utf-8")
    os.chmod(paths.config_file, 0o600)
    paths.token_file.write_text("preproduction-token\n", encoding="utf-8")
    os.chmod(paths.token_file, 0o600)
    _copy_signed_core(core_binary, paths)
    _seed_local(paths=paths, config=config, lab_config=lab_config, memory_space_id=memory_space_id)
    database = paths.resolve_rounds_database_path(config.rounds_database_path)
    core_database = paths.resolve_knowledge_database_path(config.knowledge_database_path)
    probe_report = _probe_generation_profiles(paths=paths, database=database, memory_space_id=memory_space_id)

    environment = dict(os.environ)
    python_path = os.pathsep.join(
        str(path)
        for path in (
            _LOCAL_ROOT / "src",
            _LAB_ROOT / "src",
            _INTEGRATIONS_PROTOCOL_ROOT / "src",
        )
    )
    environment["PYTHONPATH"] = python_path + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONHASHSEED"] = "0"
    log_path = work_dir / "local.log"
    process: subprocess.Popen[bytes] | None = None
    report: dict[str, Any] = {
        "started_at": _utc_now(),
        "work_dir": str(work_dir),
        "memory_space_id": memory_space_id,
        "providers": {
            "generation_model": lab_config.generation.model,
            "generation_endpoint": lab_config.generation.endpoint,
            "embedding_model": lab_config.embedding.model,
            "embedding_endpoint": lab_config.embedding.endpoint,
            "embedding_dimensions": lab_config.embedding.dimensions,
        },
        "probe": probe_report,
        "round_a": {},
        "round_b": {},
        "restart": {},
        "retrieval": {},
        "hermes": {"status": "not_run"},
        "security": {
            "core_launched_by_local": True,
            "core_database": str(core_database),
            "secret_store": str(paths.home / "secrets.json"),
        },
    }
    try:
        process = _start_local(home=home, port=port, env=environment, log_path=log_path)
        base_url = f"http://127.0.0.1:{port}"
        token = paths.token_file.read_text(encoding="utf-8").strip()

        latest_details: dict[str, Any] = {}

        def ready() -> bool:
            nonlocal latest_details
            if process is not None and process.poll() is not None:
                raise GateFailure("Local exited before full readiness")
            status, payload = _http_json(base_url, "/health/details", token=token)
            latest_details = payload
            return status == 200 and payload.get("full_ready") is True

        _wait_until(ready, timeout=timeout)
        report["readiness"] = latest_details
        if process.poll() is not None:
            raise GateFailure("Local exited after reporting readiness")

        def submit(round_payload: dict[str, Any]) -> dict[str, Any]:
            status, payload = _http_json(
                base_url,
                "/rounds",
                token=token,
                method="POST",
                payload=round_payload,
            )
            if status not in {200, 202}:
                raise GateFailure(f"Local rejected RawRound with status {status}: {payload.get('detail', payload.get('error', 'unknown'))}")
            return payload

        def wait_drained(
            *, raw_round_id: str, expected_core_counts: Mapping[str, int]
        ) -> dict[str, Any]:
            last: dict[str, Any] = {}

            def drained() -> bool:
                nonlocal last
                status, payload = _http_json(base_url, "/health/details", token=token)
                last = payload
                delivery = _local_delivery_state(database, raw_round_id)
                last["local_delivery"] = delivery
                command_status = delivery.get("command_status")
                if command_status in {"rejected", "failed"}:
                    raise GateFailure(
                        "Local-to-Core delivery failed: "
                        f"{delivery.get('command_error_code') or 'unknown'}"
                    )
                components = payload.get("components", {})
                observed_counts = _core_counts(core_database, memory_space_id)
                last["core_counts"] = observed_counts
                core_raw_round_id = delivery.get("core_raw_round_id")
                round_status = _core_round_status(core_database, core_raw_round_id)
                last["core_round_status"] = round_status
                task_state = _core_task_state(core_database)
                task_counts = _core_task_counts(core_database)
                embedding_counts = _core_embedding_counts(core_database)
                last["core_task_state"] = task_state
                last["core_task_counts"] = task_counts
                last["core_embedding_counts"] = embedding_counts
                workers = components.get("workers", {}) if isinstance(components, dict) else {}
                model_worker = workers.get("core_model_tasks", {}) if isinstance(workers, dict) else {}
                model_state = model_worker.get("state", {}) if isinstance(model_worker, dict) else {}
                if isinstance(model_state, dict) and model_state.get("last_error_code"):
                    raise GateFailure(
                        "Core model-task worker failed: "
                        f"{model_state['last_error_code']}"
                    )
                counts_match = all(
                    observed_counts.get(name) == expected
                    for name, expected in expected_core_counts.items()
                )
                if round_status in {
                    "completed",
                    "completed_with_rejections",
                    "no_values",
                    "failed",
                } and not counts_match:
                    raise GateFailure(
                        "Core round reached a terminal status before the expected "
                        f"counts: status={round_status}, counts={observed_counts}"
                    )
                return (
                    status in {200, 503}
                    and payload.get("terminal_worker_failure") is not True
                    and command_status == "completed"
                    and delivery.get("transport_status") == "accepted"
                    and task_state["pending"] == 0
                    and task_state["terminal_failures"] == 0
                    and task_counts.get("semantic_repair", 0) == 0
                    and task_counts.get("consolidate_values", 0) == 0
                    and embedding_counts.get("object_card", 0)
                    == expected_core_counts["objects"]
                    and embedding_counts.get("value", 0)
                    == expected_core_counts["active_values"]
                    and embedding_counts.get("facet", 0) >= 14
                    and counts_match
                )

            _wait_until(drained, timeout=timeout, interval=2.0)
            return last

        report["round_a"]["submission"] = submit(round_a)
        report["round_a"]["health_after_submit"] = wait_drained(
            raw_round_id=str(report["round_a"]["submission"]["raw_round_id"]),
            expected_core_counts={"objects": 3, "active_values": 4},
        )
        report["round_a"]["local_counts"] = _local_counts(database, memory_space_id)
        report["round_a"]["core_counts"] = _core_counts(core_database, memory_space_id)
        round_a_core_raw_round_id = report["round_a"]["health_after_submit"]["local_delivery"].get(
            "core_raw_round_id"
        )
        if not round_a_core_raw_round_id:
            raise GateFailure("Round A did not expose the Core raw-round identity")
        report["round_a"]["task_counts"] = _core_task_counts(
            core_database, str(round_a_core_raw_round_id)
        )
        report["round_a"]["generation_task_counts"] = _core_generation_task_counts(
            core_database, str(round_a_core_raw_round_id)
        )
        if report["round_a"]["generation_task_counts"] != {
            "extract_claims": 1,
            "resolve_subjects": 1,
        }:
            raise GateFailure(
                "Round A generation budget/task shape failed: "
                f"{report['round_a']['generation_task_counts']}"
            )

        if report["round_a"]["core_counts"]["objects"] != 3 or report["round_a"]["core_counts"]["active_values"] != 4:
            raise GateFailure(f"Round A counts failed: {report['round_a']['core_counts']}")

        _terminate(process)
        process = None
        process = _start_local(home=home, port=port, env=environment, log_path=work_dir / "local-restart.log")
        _wait_until(ready, timeout=timeout)
        report["restart"]["readiness"] = latest_details
        report["restart"]["probe_cache_hit_expected"] = True
        report["restart"]["probe"] = _probe_generation_profiles(
            paths=paths,
            database=database,
            memory_space_id=memory_space_id,
        )
        if any(
            result.get("status") != "passed" or result.get("cache_hit") is not True
            for result in report["restart"]["probe"].values()
        ):
            raise GateFailure(
                "generation capability cache was not reused after Local restart: "
                f"{report['restart']['probe']}"
            )
        report["restart"]["capability_cache_rows"] = len(
            DatabaseBackedCapabilityStore(str(database)).list_capabilities()
        )

        report["round_b"]["submission"] = submit(round_b)
        report["round_b"]["health_after_submit"] = wait_drained(
            raw_round_id=str(report["round_b"]["submission"]["raw_round_id"]),
            expected_core_counts={"objects": 3, "active_values": 6},
        )
        report["round_b"]["local_counts"] = _local_counts(database, memory_space_id)
        report["round_b"]["core_counts"] = _core_counts(core_database, memory_space_id)
        round_b_core_raw_round_id = report["round_b"]["health_after_submit"]["local_delivery"].get(
            "core_raw_round_id"
        )
        if not round_b_core_raw_round_id:
            raise GateFailure("Round B did not expose the Core raw-round identity")
        report["round_b"]["task_counts"] = _core_task_counts(
            core_database, str(round_b_core_raw_round_id)
        )
        report["round_b"]["generation_task_counts"] = _core_generation_task_counts(
            core_database, str(round_b_core_raw_round_id)
        )
        if report["round_b"]["generation_task_counts"] != {
            "extract_claims": 1,
            "resolve_subjects": 1,
        }:
            raise GateFailure(
                "Round B generation budget/task shape failed: "
                f"{report['round_b']['generation_task_counts']}"
            )
        if report["round_b"]["core_counts"]["objects"] != 3 or report["round_b"]["core_counts"]["active_values"] != 6:
            raise GateFailure(f"Round B counts failed: {report['round_b']['core_counts']}")
        report["round_b"]["control_findings"] = int(
            (report["round_b"]["health_after_submit"].get("components", {})
             .get("object_facet", {})
             .get("integrity_finding_count", 0))
            or 0
        )
        if report["round_b"]["control_findings"] != 0:
            raise GateFailure(
                "Round B control findings are not clean: "
                f"{report['round_b']['control_findings']}"
            )

        status, retrieval = _http_json(
            base_url,
            "/context/retrieve",
            token=token,
            method="POST",
            payload={
                "schema_version": 2,
                "memory_space_id": memory_space_id,
                "query": "Как работает выбор страны и языка на сайте?",
                "limit": 5,
            },
        )
        if status != 200:
            raise GateFailure(f"Local retrieval failed with status {status}")
        report["retrieval"] = retrieval
        if not retrieval.get("items"):
            raise GateFailure("production retrieval returned no ContextView items")

        # Hermes is exercised through its real runtime and the same Local API.
        from ledgermind_integrations.adapters.hermes.config import HermesConfig  # noqa: PLC0415
        from ledgermind_integrations.adapters.hermes.runtime import HermesPluginRuntime  # noqa: PLC0415
        from ledgermind_integrations.runtime.client import LedgerMindClient  # noqa: PLC0415
        from ledgermind_integrations.runtime.spool import FileSpool  # noqa: PLC0415

        hermes_config = HermesConfig(
            endpoint=base_url,
            token_file=str(paths.token_file),
            memory_space_id=memory_space_id,
            source_instance_id="hermes-preproduction-gate",
            profile_id="default",
            state_db_path=str(work_dir / "hermes-state.db"),
            spool_dir=str(work_dir / "hermes-spool"),
            context_timeout_seconds=10.0,
            request_timeout_seconds=10.0,
            worker_poll_seconds=0.05,
        )
        hermes_client = LedgerMindClient(
            endpoint=base_url,
            token_file=paths.token_file,
            timeout=10.0,
        )
        hermes_runtime = HermesPluginRuntime(
            config=hermes_config,
            client=hermes_client,
            spool=FileSpool(hermes_config.spool_dir),
        )
        try:
            hermes_runtime.start()
            context = hermes_runtime.on_pre_llm_call(
                session_id="hermes-production-session",
                turn_id="round-c",
                user_message="Как связаны язык интерфейса и переключатель страны?",
            )
            if not context or not context.get("context"):
                raise GateFailure("Hermes did not receive ledgermind_context")
            hermes_runtime.on_post_llm_call(
                session_id="hermes-production-session",
                turn_id="round-c",
                user_message="Как связаны язык интерфейса и переключатель страны?",
                assistant_response="Проверю сохранённые правила интерфейса.",
            )
            _wait_until(
                lambda: any(
                    path.name.endswith(".json")
                    for path in Path(hermes_config.spool_dir).glob("**/*")
                    if path.is_file()
                ),
                timeout=10.0,
                interval=0.2,
            )
            report["hermes"] = {
                "status": "passed",
                "context_injected": True,
                "captured_round": True,
                "self_echo_extensions": True,
            }
        finally:
            hermes_runtime.shutdown()
    finally:
        _terminate(process)
        report["completed_at"] = _utc_now()
        if work_context is not None:
            work_context.cleanup()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-config", type=Path, required=True)
    parser.add_argument(
        "--core-binary",
        type=Path,
        default=_WORKSPACE_ROOT / "ledgermind-core" / "target" / "release" / "ledgermind-core",
    )
    parser.add_argument("--lab-root", type=Path, default=_LAB_ROOT)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-work-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_gate(
            lab_config_path=args.lab_config.expanduser().resolve(),
            core_binary=args.core_binary.expanduser().resolve(),
            lab_root=args.lab_root.expanduser().resolve(),
            timeout=max(args.timeout, 30.0),
            keep_work_dir=args.keep_work_dir.expanduser().resolve() if args.keep_work_dir else None,
        )
        if args.report:
            args.report.expanduser().resolve().write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must return a bounded gate result
        payload = {
            "status": "failed",
            "error_code": type(exc).__name__,
            "error": str(exc)[:500],
        }
        if args.report:
            args.report.expanduser().resolve().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
