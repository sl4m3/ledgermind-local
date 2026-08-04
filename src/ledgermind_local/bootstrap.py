"""Bootstrap utilities for the Local service and its Rust Core boundary."""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ledgermind_local.config import LocalConfig, WorkerConfig
from ledgermind_local.core_gateway import (
    ContextViewResult,
    CoreGateway,
    ProcessCoreGateway,
    RetrieveContextCommand,
)
from ledgermind_local.core_gateway.isolation import IsolationRequirements
from ledgermind_local.core_gateway.signing import verify_core_binary
from ledgermind_local.core_gateway.supervisor import CoreSupervisor
from ledgermind_local.inference import InferenceBroker, SecretStore
from ledgermind_local.maintenance.core_backup import CoreBackupService
from ledgermind_local.paths import ServicePaths
from ledgermind_local.persistence import open_sqlite_connection
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.processing import (
    BrokerHypothesisGenerator,
    HypothesisGenerator,
    RoundProcessingWorker,
)
from ledgermind_local.raw_rounds import RawRoundIngestHandler
from ledgermind_local.scheduler import (
    CoreCommandWorker,
    CoreModelTaskWorker,
    CoreProjectionWorker,
    GuardedWorkerLoop,
    RawRoundRetentionWorker,
)
from ledgermind_local.scheduler.worker_state import WorkerState, WorkerStateSnapshot
from ledgermind_local.search import CoreBackedSearch
from ledgermind_local.search.fts import CoreProjectionSearchAdapter
from ledgermind_local.search.vector import CoreProjectionVectorSearchAdapter
from ledgermind_local.service_lock import ServiceLock

logger = logging.getLogger(__name__)

DEFAULT_PROJECTIONS: tuple[str, ...] = (
    "projections.search",
    "projections.knowledge",
    "projections.markdown",
)


def build_projection_names(config: LocalConfig) -> tuple[str, ...]:
    names = list(DEFAULT_PROJECTIONS)
    if config.markdown_projection.enabled or config.markdown_audit_enabled:
        names.append("projections.markdown_audit")
    return tuple(names)


def build_core_projection_handlers(
    *,
    connection: sqlite3.Connection,
    database_path: str | Path,
    config: LocalConfig,
) -> dict[str, Any]:
    """Build Local-only handlers for public Rust Core projection events."""

    from ledgermind_local.projections import (
        KnowledgeFTSProjection,
        KnowledgeMarkdownProjection,
        KnowledgeVectorProjection,
    )

    handlers: dict[str, Any] = {
        "projections.search": KnowledgeFTSProjection(connection),
    }
    if config.vector.enabled:
        model_path = Path(config.vector.model_path).expanduser()
        from ledgermind_local.projections import GGUFVectorizer

        handlers["projections.knowledge"] = KnowledgeVectorProjection(
            connection=connection,
            vector_store_root=_build_vector_store_root(database_path),
            vectorizer_factory=lambda: GGUFVectorizer(
                model_path=model_path,
                gpu_layers=config.vector.gpu_layers,
            ),
        )
    if config.markdown_projection.enabled:
        handlers["projections.markdown"] = KnowledgeMarkdownProjection(
            connection=connection,
            markdown_root=_build_markdown_root(database_path),
        )
    return handlers


def _build_vector_store_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".vectors")


def _build_markdown_root(database_path: str | Path) -> Path:
    return Path(database_path).with_suffix(".markdown")


def _build_vectorizer_factory() -> Callable[[], Any] | None:
    model_path = os.environ.get("LEDGERMIND_VECTOR_MODEL_PATH")
    if not model_path:
        return None

    from ledgermind_local.projections import GGUFVectorizer

    return lambda: GGUFVectorizer(model_path=model_path)


def build_ingest_raw_round_handler(
    *,
    database_path: str | Path,
    max_raw_round_bytes: int = 5_000_000,
    retention_days: int = 30,
    pipeline_version: int = 1,
    normalizer_version: int = 1,
    prompt_version: int = 1,
) -> RawRoundIngestHandler:
    """Build the Local-owned RawRound capture handler."""

    return RawRoundIngestHandler(
        database_path=database_path,
        max_raw_round_bytes=max_raw_round_bytes,
        retention_days=retention_days,
        pipeline_version=pipeline_version,
        normalizer_version=normalizer_version,
        prompt_version=prompt_version,
    )


def build_round_processing_worker(
    *,
    database_path: str | Path,
    generator: HypothesisGenerator | None = None,
    broker: InferenceBroker | None = None,
    hypothesis_profile_id: str | None = None,
    secrets_path: str | Path | None = None,
    worker_id: str | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 30,
    lease_seconds: float = 300,
    heartbeat_interval_seconds: float = 30,
) -> RoundProcessingWorker:
    """Build Local hypothesis generation and durable Core-command worker."""

    if generator is not None:
        selected_generator = generator
    else:
        if not hypothesis_profile_id:
            raise ValueError(
                "hypothesis_profile_id is required when processing uses the inference broker"
            )
        selected_broker = broker
        if selected_broker is None:
            resolved_secrets_path = (
                Path(secrets_path)
                if secrets_path is not None
                else Path(database_path).parent / "secrets.json"
            )
            selected_broker = InferenceBroker(
                database_path=database_path,
                secret_store=SecretStore(resolved_secrets_path),
            )
        profile = selected_broker.get_profile(hypothesis_profile_id)
        selected_generator = BrokerHypothesisGenerator(
            selected_broker,
            profile_id=hypothesis_profile_id,
            provider=profile.provider_kind,
            model=profile.model,
            prompt_version=profile.hypothesis_prompt_version,
            schema_version=profile.hypothesis_schema_version,
        )
    return RoundProcessingWorker(
        database_path=database_path,
        generator=selected_generator,
        worker_id=worker_id,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


@dataclass(slots=True)
class _RuntimeWorkerHandle:
    """One worker, its guarded loop and its lifecycle state."""

    name: str
    config: WorkerConfig
    worker: object
    loop: GuardedWorkerLoop
    state: WorkerState


class _RuntimeProjectionCandidates:
    """Open projection connections per request so API threads share no SQLite handle."""

    def __init__(
        self,
        *,
        database_path: Path,
        vector: bool = False,
        vector_store_root: Path | None = None,
        vectorizer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.database_path = database_path
        self.vector = vector
        self.vector_store_root = vector_store_root
        self.vectorizer_factory = vectorizer_factory

    def search(self, memory_space_id: str, query: str, limit: int) -> object:
        connection = open_sqlite_connection(self.database_path)
        adapter: Any
        try:
            if self.vector:
                if self.vector_store_root is None or self.vectorizer_factory is None:
                    raise RuntimeError("vector search is not configured")
                adapter = CoreProjectionVectorSearchAdapter(
                    connection=connection,
                    vector_store_root=self.vector_store_root,
                    vectorizer_factory=self.vectorizer_factory,
                )
            else:
                adapter = CoreProjectionSearchAdapter(connection)
            return adapter.search(memory_space_id, query, limit)
        finally:
            close = locals().get("adapter")
            if close is not None:
                close_method = getattr(close, "close", None)
                if callable(close_method):
                    close_method()
            connection.close()


class _RuntimeCoreBackedSearch(CoreBackedSearch):
    """Apply B2 search settings without changing the accepted A4 search port."""

    def __init__(
        self,
        *,
        candidate_multiplier: int,
        fallback_to_core_fts: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._candidate_multiplier = max(int(candidate_multiplier), 1)
        self._fallback_to_core_fts = fallback_to_core_fts

    def retrieve_context(
        self,
        request: RetrieveContextCommand | None = None,
        *,
        request_id: str | None = None,
        memory_space_id: str | None = None,
        query: str | None = None,
        limit: int | None = None,
        candidate_limit: int | None = None,
    ) -> ContextViewResult:
        effective_limit = candidate_limit
        if effective_limit is None:
            request_limit = request.limit if request is not None else limit
            if request_limit is not None:
                effective_limit = max(int(request_limit), 1) * self._candidate_multiplier
        if request is not None:
            return super().retrieve_context(request, candidate_limit=effective_limit)
        return super().retrieve_context(
            request_id=request_id,
            memory_space_id=memory_space_id,
            query=query,
            limit=limit,
            candidate_limit=effective_limit,
        )

    def _mark_degraded(self) -> None:
        super()._mark_degraded()
        if not self._fallback_to_core_fts:
            raise RuntimeError("local candidate search is unavailable")


def _core_isolation_requirements(config: LocalConfig) -> IsolationRequirements:
    security = config.core_security
    return IsolationRequirements(
        require_network_isolation=(
            security.require_network_isolation
            or config.require_core_network_isolation
        ),
        require_rounds_database_hidden=security.require_rounds_database_hidden,
        require_filesystem_allowlist=security.require_filesystem_allowlist,
        require_environment_sanitized=security.require_environment_sanitized,
        require_signature=security.require_signature,
    )


def build_process_core_gateway(*, paths: ServicePaths, config: LocalConfig) -> ProcessCoreGateway:
    """Build the process Core boundary without starting the child process."""

    command_path = paths.resolve_core_path(config.core_binary_path)
    signature_path = paths.resolve_core_path(config.core_signature_path)
    public_key_path = paths.resolve_core_path(config.core_public_key_path)
    knowledge_database_path = paths.resolve_knowledge_database_path(
        config.knowledge_database_path
    )
    paths.core_data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    signature_verified = False
    if config.verify_core_signature:
        verify_core_binary(
            command_path,
            signature_path=signature_path,
            public_key_path=public_key_path,
        )
        signature_verified = True
    requirements = _core_isolation_requirements(config)
    supervisor = CoreSupervisor(
        [str(command_path), "--database", str(knowledge_database_path)],
        startup_timeout_seconds=config.core_startup_timeout_seconds,
        operation_timeout_seconds=config.core_request_timeout_seconds,
        core_data_dir=paths.core_data_dir,
        blocked_data_dirs=(paths.home,),
        rounds_database_path=paths.resolve_rounds_database_path(
            config.rounds_database_path
        ),
        runtime_paths=(paths.core_data_dir,),
        isolation_requirements=requirements,
        strict_isolation=(
            config.core_security.profile == "secure"
            or any(requirements.as_dict().values())
        ),
        binary_signature_verified=signature_verified,
    )
    return ProcessCoreGateway(supervisor)


class LocalRuntime:
    """Own the Local service lifecycle from migration to shutdown.

    Core is deliberately optional for capture: a failed Core handshake, missing
    daemon or unavailable secure sandbox leaves the Local RawRound writer alive,
    while full readiness remains false and Core-dependent workers are not started.
    """

    _WORKER_ORDER = (
        "retention",
        "processing",
        "core_commands",
        "core_projections",
        "core_model_tasks",
    )

    def __init__(
        self,
        *,
        paths: ServicePaths,
        config: LocalConfig,
        api_token: str | None,
        database_path: str | Path | None = None,
        core_gateway: CoreGateway | None = None,
        core_gateway_factory: Callable[[], CoreGateway] | None = None,
        migration_runner: Callable[[sqlite3.Connection], object] | None = None,
        connection_factory: Callable[[str | Path], sqlite3.Connection] = open_sqlite_connection,
        lock_factory: Callable[[Path], ServiceLock] = ServiceLock,
        pid_writer: Callable[[Path, int], None] | None = None,
        pid_remover: Callable[[Path], None] | None = None,
        worker_factories: Mapping[str, Callable[[LocalRuntime], object]] | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.api_token = api_token
        self.database_path = Path(database_path or paths.resolve_rounds_database_path(
            config.rounds_database_path
        )).expanduser()
        self.core_gateway = core_gateway
        self.core_gateway_factory = core_gateway_factory or (
            lambda: build_process_core_gateway(paths=self.paths, config=self.config)
        )
        self.migration_runner = migration_runner or migrations.apply_migrations
        self.connection_factory = connection_factory
        self.lock_factory = lock_factory
        self.pid_writer = pid_writer or _write_runtime_pid
        self.pid_remover = pid_remover or _remove_runtime_pid
        self.worker_factories = dict(worker_factories or {})
        self._service_lock: ServiceLock | None = None
        self._lock_acquired = False
        self._pid_owned = False
        self._started = False
        self._starting = False
        self._stop_requested = False
        self._migrations_applied = False
        self._core_ready = False
        self._core_error_code: str | None = None
        self._component_errors: dict[str, str] = {}
        self._workers: dict[str, _RuntimeWorkerHandle] = {}
        self._prepared_workers: dict[str, object] = {}
        self._raw_round_handler: RawRoundIngestHandler | None = None
        self._context_search: object | None = None
        self._backup_service: CoreBackupService | None = None
        self._degraded_search = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def capture_ready(self) -> bool:
        return bool(
            self._started
            and self._migrations_applied
            and self._service_lock is not None
            and self._lock_acquired
            and self._raw_round_handler is not None
            and callable(getattr(self._raw_round_handler, "handle", None))
        )

    @property
    def full_ready(self) -> bool:
        report = self.health_report()
        return bool(report["full_ready"])

    @property
    def context_search(self) -> object | None:
        return self._context_search

    @property
    def ingest_handler(self) -> RawRoundIngestHandler | None:
        """B2 composition name for the Local-owned capture handler."""

        return self._raw_round_handler

    @property
    def worker_loops(self) -> dict[str, GuardedWorkerLoop]:
        return {name: handle.loop for name, handle in self._workers.items()}

    @property
    def retention_worker(self) -> object | None:
        handle = self._workers.get("retention")
        return handle.worker if handle is not None else None

    @property
    def backup_service(self) -> object | None:
        return getattr(self, "_backup_service", None)

    @property
    def worker_states(self) -> dict[str, WorkerStateSnapshot]:
        return {name: handle.state.snapshot() for name, handle in self._workers.items()}

    def worker_state(self, name: str) -> WorkerStateSnapshot | None:
        handle = self._workers.get(name)
        return handle.state.snapshot() if handle is not None else None

    def build_ingest_raw_round_handler(self) -> RawRoundIngestHandler:
        if self._raw_round_handler is None:
            self._raw_round_handler = build_ingest_raw_round_handler(
                database_path=self.database_path,
                max_raw_round_bytes=self.config.max_raw_round_bytes,
                retention_days=self.config.raw_round_retention_days,
            )
        return self._raw_round_handler

    def start(self) -> LocalRuntime:
        """Acquire ownership, migrate, then start Core and guarded workers."""

        if self._started:
            return self
        if self._starting:
            raise RuntimeError("Local runtime startup is already in progress")
        self._starting = True
        self._stop_requested = False
        self._component_errors.clear()
        self._workers.clear()
        self._core_error_code = None
        self._core_ready = False
        self._degraded_search = False
        self._migrations_applied = False
        self._context_search = None
        self._backup_service = None
        self._raw_round_handler = None
        if self.config.core_security.profile == "permissive":
            logger.warning(
                "permissive Core security profile is enabled; isolation guarantees are not enforced"
            )
        try:
            self.paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._service_lock = self.lock_factory(self.paths.service_lock_file)
            self._acquire_service_lock()
            self._run_migrations_before_workers()
            self._raw_round_handler = build_ingest_raw_round_handler(
                database_path=self.database_path,
                max_raw_round_bytes=self.config.max_raw_round_bytes,
                retention_days=self.config.raw_round_retention_days,
            )
            self.pid_writer(self.paths.service_pid_file, os.getpid())
            self._pid_owned = True
            # Core is brought up only after Local's durable schema is ready;
            # no worker can observe a partially migrated database.
            self._start_core_if_available()
            self._build_context_search()
            self._start_workers()
            self._started = True
            return self
        except BaseException:
            self._cleanup_after_failed_start()
            raise
        finally:
            self._starting = False

    def request_stop(self) -> None:
        """Signal all workers without waiting; used by HTTP server signal hooks."""

        self._stop_requested = True
        for handle in reversed(tuple(self._workers.values())):
            handle.loop.request_stop()

    def stop(self) -> None:
        """Stop workers in reverse composition order, then Core and ownership files."""

        if not self._started and self._service_lock is None and not self._pid_owned:
            return
        self.request_stop()
        for name in reversed(self._WORKER_ORDER):
            handle = self._workers.get(name)
            if handle is None:
                continue
            stopped = handle.loop.join(handle.config.shutdown_timeout_seconds)
            if not stopped:
                self._component_errors[name] = "shutdown_timeout"
        self._workers.clear()
        self._context_search = None
        self._backup_service = None
        self._raw_round_handler = None
        gateway = self.core_gateway
        self.core_gateway = None
        if gateway is not None:
            close = getattr(gateway, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                    self._component_errors["core"] = _safe_error_code(exc)
        if self._pid_owned:
            try:
                self.pid_remover(self.paths.service_pid_file)
            finally:
                self._pid_owned = False
        if self._service_lock is not None:
            self._release_service_lock()
            self._service_lock = None
        self._started = False
        self._stop_requested = False

    def health_report(self) -> dict[str, object]:
        """Return secret-free capture/full readiness and component diagnostics."""

        worker_reports: dict[str, object] = {}
        workers_ready = True
        for name, config_worker in self._configured_workers():
            handle = self._workers.get(name)
            if not config_worker.enabled:
                worker_reports[name] = {"enabled": False, "ready": True}
                continue
            if handle is None:
                error_code = self._component_errors.get(name, "not_started")
                worker_reports[name] = {
                    "enabled": True,
                    "ready": False,
                    "error_code": error_code,
                }
                workers_ready = False
                continue
            state = handle.state.snapshot()
            loop_thread = getattr(handle.loop, "_thread", None)
            alive = bool(loop_thread is not None and loop_thread.is_alive())
            ready = bool(alive and state.healthy and name not in self._component_errors)
            worker_reports[name] = {
                "enabled": True,
                "ready": ready,
                "state": {
                    "running": state.running,
                    "healthy": state.healthy,
                    "last_error_code": state.last_error_code,
                    "consecutive_failures": state.consecutive_failures,
                    "processed_count": state.processed_count,
                    "failed_count": state.failed_count,
                },
            }
            workers_ready = workers_ready and ready

        isolation = self._isolation_report()
        core_report = {
            "ready": self._core_ready,
            "available": self.core_gateway is not None,
            "error_code": self._core_error_code,
            "isolation": isolation,
        }
        capture_ready = self.capture_ready
        core_security_ready = bool(isolation.get("ready", True))
        inference_ready = not bool(self._component_errors.get("processing"))
        retention_report = worker_reports.get("retention", {"enabled": False, "ready": True})
        projections_report = worker_reports.get(
            "core_projections", {"enabled": False, "ready": True}
        )
        isolation_report = dict(isolation)
        capabilities_payload = isolation_report.get("capabilities")
        if isinstance(capabilities_payload, dict):
            capabilities_payload = dict(capabilities_payload)
            capabilities_payload.pop("detail", None)
            isolation_report["capabilities"] = capabilities_payload
        full_ready = bool(
            capture_ready
            and self._core_ready
            and core_security_ready
            and workers_ready
            and inference_ready
        )
        core_report["isolation"] = isolation_report
        projection_ready = bool(
            projections_report.get("ready", True)
            if isinstance(projections_report, dict)
            else True
        )
        return {
            "status": "ready" if full_ready else ("capture-ready" if capture_ready else "unavailable"),
            "capture_ready": capture_ready,
            "full_ready": full_ready,
            "components": {
                "capture": {
                    "ready": capture_ready,
                    "migrations_applied": self._migrations_applied,
                    "service_lock_held": self._lock_acquired,
                    "raw_round_writer": self._raw_round_handler is not None,
                },
                "core": core_report,
                "isolation": isolation_report,
                "inference": {
                    "ready": inference_ready,
                    "processing_enabled": self.config.workers.processing.enabled,
                },
                "workers": worker_reports,
                "projections": {"ready": projection_ready},
                "retention": retention_report,
                "search": {"degraded": self._degraded_search},
            },
            "workers": worker_reports,
        }

    def _acquire_service_lock(self) -> None:
        if self._service_lock is None:
            raise RuntimeError("service lock is not configured")
        acquire = getattr(self._service_lock, "acquire", None)
        if callable(acquire):
            acquire()
        else:
            enter = getattr(self._service_lock, "__enter__", None)
            if not callable(enter):
                raise TypeError("service lock does not implement acquire()")
            enter()
        self._lock_acquired = True

    def _release_service_lock(self) -> None:
        if self._service_lock is None or not self._lock_acquired:
            return
        release = getattr(self._service_lock, "release", None)
        if callable(release):
            release()
        else:
            exit_method = getattr(self._service_lock, "__exit__", None)
            if callable(exit_method):
                exit_method(None, None, None)
        self._lock_acquired = False

    def _run_migrations_before_workers(self) -> None:
        connection = self.connection_factory(self.database_path)
        entered = False
        try:
            enter = getattr(connection, "__enter__", None)
            if callable(enter):
                connection = enter()
                entered = True
            self.migration_runner(connection)
            if getattr(connection, "in_transaction", False):
                connection.commit()
            execute = getattr(connection, "execute", None)
            if callable(execute):
                result = execute("PRAGMA integrity_check").fetchone()
                if result is not None and result[0] != "ok":
                    raise RuntimeError("Local database integrity check failed")
            self._migrations_applied = True
        finally:
            if entered:
                exit_method = getattr(connection, "__exit__", None)
                if callable(exit_method):
                    exit_method(None, None, None)
            else:
                close = getattr(connection, "close", None)
                if callable(close):
                    close()

    def _start_core_if_available(self) -> None:
        try:
            if self.core_gateway is None:
                self.core_gateway = self.core_gateway_factory()
            if self.core_gateway is None:
                raise RuntimeError("Core gateway is unavailable")
            required = ["base", "maintenance"]
            workers = self.config.workers
            if workers.core_projections.enabled:
                required.append("projections")
            if workers.core_model_tasks.enabled:
                required.append("model_tasks")
            start = getattr(self.core_gateway, "start", None)
            if callable(start):
                start()
            require_capabilities = getattr(self.core_gateway, "require_capabilities", None)
            if callable(require_capabilities):
                require_capabilities(*required)
            health = getattr(self.core_gateway, "health", None)
            if callable(health):
                health_result = health()
                self._core_ready = bool(
                    getattr(health_result, "healthy", health_result is True)
                )
            else:
                self._core_ready = True
            if not self._core_ready:
                raise RuntimeError("Core health check failed")
            self._backup_service = CoreBackupService(
                gateway=self.core_gateway,
                core_data_dir=self.paths.core_data_dir,
                rounds_database_path=self.database_path,
            )
        except Exception as exc:  # noqa: BLE001 - capture remains available
            self._core_ready = False
            self._core_error_code = _safe_error_code(exc)
            gateway = self.core_gateway
            if gateway is not None:
                close = getattr(gateway, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.debug("failed to close Core gateway", exc_info=True)
            self.core_gateway = None

    def _start_workers(self) -> None:
        for name in self._WORKER_ORDER:
            config_worker = getattr(self.config.workers, name)
            if not config_worker.enabled:
                continue
            if name in {"core_commands", "core_projections", "core_model_tasks"} and not self._core_ready:
                self._component_errors[name] = "core_unavailable"
                continue
            try:
                worker = self._build_worker(name)
                handle = self._start_worker_loop(name, config_worker, worker)
                self._workers[name] = handle
            except Exception as exc:  # noqa: BLE001 - one worker cannot kill capture
                self._component_errors[name] = _safe_error_code(exc)
                logger.warning(
                    "worker startup failed",
                    extra={"worker": name, "error_code": _safe_error_code(exc)},
                )

    def _build_worker(self, name: str) -> object:
        factory = self.worker_factories.get(name)
        if factory is not None:
            return factory(self)
        if name == "retention":
            return RawRoundRetentionWorker(self.database_path)
        if name == "processing":
            if self.config.hypothesis_profile_id is None:
                raise ValueError("processing_enabled requires hypothesis_profile_id")
            return build_round_processing_worker(
                database_path=self.database_path,
                hypothesis_profile_id=self.config.hypothesis_profile_id,
                secrets_path=self.paths.resolve_home_path(
                    self.config.inference_secrets_path
                ),
                worker_id=f"processing:{os.getpid()}",
                max_attempts=self.config.processing_max_attempts,
                retry_delay_seconds=self.config.processing_retry_delay_seconds,
                lease_seconds=self.config.processing_lease_seconds,
                heartbeat_interval_seconds=self.config.processing_heartbeat_interval_seconds,
            )
        if self.core_gateway is None:
            raise RuntimeError("Core gateway is unavailable")
        if name == "core_commands":
            return CoreCommandWorker(
                database_path=self.database_path,
                gateway=self.core_gateway,
                worker_id=f"core-command:{os.getpid()}",
                max_attempts=self.config.processing_max_attempts,
                retry_delay_seconds=self.config.processing_retry_delay_seconds,
                lease_seconds=self.config.processing_lease_seconds,
                state=WorkerState(name),
            )
        if name == "core_projections":
            return CoreProjectionWorker(
                database_path=self.database_path,
                gateway=self.core_gateway,
                consumer_id="local-projections",
                handlers_factory=lambda connection: build_core_projection_handlers(
                    connection=connection,
                    database_path=self.database_path,
                    config=self.config,
                ),
                state=WorkerState(name),
            )
        if name == "core_model_tasks":
            broker = InferenceBroker(
                database_path=self.database_path,
                secret_store=SecretStore(
                    self.paths.resolve_home_path(self.config.inference_secrets_path)
                ),
            )
            return CoreModelTaskWorker(
                database_path=self.database_path,
                gateway=self.core_gateway,
                broker=broker,
                worker_id="local-model-tasks",
                lease_seconds=int(self.config.processing_lease_seconds),
            )
        raise ValueError(f"unknown Local worker: {name}")

    def _start_worker_loop(
        self,
        name: str,
        config_worker: WorkerConfig,
        worker: object,
    ) -> _RuntimeWorkerHandle:
        state = getattr(worker, "state", None)
        if not isinstance(state, WorkerState):
            state = WorkerState(name)
        create_loop = getattr(worker, "create_loop", None)
        if callable(create_loop):
            loop = create_loop(
                poll_interval_seconds=config_worker.interval_seconds,
                max_backoff_seconds=config_worker.max_backoff_seconds,
            )
        else:
            loop = GuardedWorkerLoop(
                worker,  # type: ignore[arg-type]
                state=state,
                name=name,
                poll_interval_seconds=config_worker.interval_seconds,
                max_backoff_seconds=config_worker.max_backoff_seconds,
                close_on_stop=callable(getattr(worker, "close", None)),
            )
        if not isinstance(loop, GuardedWorkerLoop):
            raise TypeError("worker factory returned an unsupported loop")
        loop.start()
        return _RuntimeWorkerHandle(name, config_worker, worker, loop, state)

    def _build_context_search(self) -> None:
        if self.core_gateway is None:
            self._context_search = None
            return
        if not self.config.search.enabled:
            self._context_search = self.core_gateway
            return
        fts = _RuntimeProjectionCandidates(database_path=self.database_path)
        vector = None
        if self.config.vector.enabled:
            from ledgermind_local.projections import GGUFVectorizer

            model_path = Path(self.config.vector.model_path).expanduser()
            vector = _RuntimeProjectionCandidates(
                database_path=self.database_path,
                vector=True,
                vector_store_root=_build_vector_store_root(self.database_path),
                vectorizer_factory=lambda: GGUFVectorizer(
                    model_path=model_path,
                    gpu_layers=self.config.vector.gpu_layers,
                ),
            )
        self._context_search = _RuntimeCoreBackedSearch(
            fts_search=fts,
            vector_search=vector,
            core_gateway=self.core_gateway,
            status_callback=self._mark_search_degraded,
            candidate_multiplier=self.config.search.candidate_multiplier,
            fallback_to_core_fts=self.config.search.fallback_to_core_fts,
        )

    def _mark_search_degraded(self, status: str) -> None:
        if status == "degraded":
            self._degraded_search = True

    def _configured_workers(self) -> tuple[tuple[str, WorkerConfig], ...]:
        return tuple(
            (name, getattr(self.config.workers, name)) for name in self._WORKER_ORDER
        )

    def _isolation_report(self) -> dict[str, object]:
        gateway = self.core_gateway
        if gateway is None:
            return {"ready": False, "missing": (), "capabilities": {}}
        capabilities = getattr(gateway, "isolation_capabilities", None)
        if capabilities is None:
            return {"ready": True, "missing": (), "capabilities": {}}
        as_dict = getattr(capabilities, "as_dict", None)
        payload = as_dict() if callable(as_dict) else {}
        missing = _core_isolation_requirements(self.config).missing(capabilities)
        return {"ready": not missing, "missing": missing, "capabilities": payload}

    def _cleanup_after_failed_start(self) -> None:
        self.request_stop()
        for handle in reversed(tuple(self._workers.values())):
            handle.loop.join(handle.config.shutdown_timeout_seconds)
        self._workers.clear()
        gateway = self.core_gateway
        self.core_gateway = None
        self._backup_service = None
        if gateway is not None:
            close = getattr(gateway, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("failed to close Core gateway", exc_info=True)
        if self._pid_owned:
            try:
                self.pid_remover(self.paths.service_pid_file)
            finally:
                self._pid_owned = False
        if self._service_lock is not None:
            self._release_service_lock()
            self._service_lock = None
        self._started = False


def _safe_error_code(error: BaseException) -> str:
    return type(error).__name__ or "RuntimeError"


def _write_runtime_pid(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = pid_file.with_name(f".{pid_file.name}.{pid}.tmp")
    try:
        temporary.write_text(f"{pid}\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, pid_file)
        os.chmod(pid_file, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_runtime_pid(pid_file: Path) -> None:
    try:
        content = pid_file.read_text(encoding="utf-8").strip()
        if int(content) != os.getpid():
            return
    except (FileNotFoundError, ValueError, OSError):
        return
    pid_file.unlink(missing_ok=True)
def bootstrap_local_service(
    *,
    home: str | Path = "~/.ledgermind/local",
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig]:
    """Prepare Local filesystem and return resolved runtime objects."""

    paths = ServicePaths(home=home)
    cfg = config or LocalConfig(config_version=1)
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    database_path = paths.resolve_rounds_database_path(cfg.rounds_database_path)
    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(database_path)
    connection.close()
    return paths, cfg


def _atomic_write_text(path: Path, data: str, *, mode: int) -> None:
    """Write text atomically with explicit permissions."""

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
            os.chmod(tmp_path, mode)
            tmp_path.replace(path)
            os.chmod(path, mode)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def initialize_local_layout(
    *,
    home: str | Path = "~/.ledgermind/local",
    force: bool = False,
    rotate_token: bool = False,
    config: LocalConfig | None = None,
) -> tuple[ServicePaths, LocalConfig, str]:
    """Initialize Local directories, config and API token."""

    preloaded_config = config
    if preloaded_config is None and not force:
        candidate = ServicePaths(home=home).config_file
        if candidate.exists():
            preloaded_config = LocalConfig.from_file(candidate)
    paths, cfg = bootstrap_local_service(home=home, config=preloaded_config)

    config_path = paths.config_file
    if force or not config_path.exists():
        cfg = config or LocalConfig(config_version=1)
        _atomic_write_text(config_path, cfg.to_json(), mode=0o600)
    elif config is None:
        cfg = LocalConfig.from_file(config_path)
    else:
        cfg = config

    database_path = paths.resolve_rounds_database_path(cfg.rounds_database_path)
    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = open_sqlite_connection(database_path)
    connection.close()

    if not paths.token_file.exists() or (force and rotate_token):
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    if rotate_token:
        token = _generate_token()
        _atomic_write_text(paths.token_file, token, mode=0o600)
        return paths, cfg, token

    existing_token = paths.token_file.read_text(encoding="utf-8")
    return paths, cfg, existing_token