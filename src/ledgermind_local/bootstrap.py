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

from ledgermind_local.config import CURRENT_CONFIG_VERSION, LocalConfig, WorkerConfig
from ledgermind_local.core_gateway import (
    ContextViewResult,
    CoreGateway,
    ProcessCoreGateway,
    RecordRetrievalOutcomeV2Command,
    RetrieveContextCommand,
    RetrieveContextV2Command,
    RetrieveContextV2Result,
)
from ledgermind_local.core_gateway.security_policy import (
    build_core_isolation_requirements,
)
from ledgermind_local.core_gateway.signing import verify_core_binary
from ledgermind_local.core_gateway.supervisor import CoreSupervisor
from ledgermind_local.inference import InferenceBroker, SecretStore
from ledgermind_local.inference.core_task_executor import CoreTaskExecutor
from ledgermind_local.inference.embedding_provider import EmbeddingProvider
from ledgermind_local.inference.profile_slots import (
    DatabaseBackedProfileResolver,
    ProfileSlot,
)
from ledgermind_local.inference.structured_json_provider import StructuredJsonProvider
from ledgermind_local.maintenance.coordinated_restore import (
    CoordinatedRestoreError,
    CoordinatedRestoreService,
)
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
    CoreExecutionTaskWorker,
    CoreModelTaskWorkerStats,
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


def _build_runtime_vectorizer_factory(config: LocalConfig) -> Callable[[], Any]:
    """Build the technical embedding backend used by generic Core tasks."""

    if not config.vector.enabled:
        def unavailable_vectorizer() -> Any:
            raise RuntimeError("local embedding backend is disabled")

        return unavailable_vectorizer

    from ledgermind_local.projections import GGUFVectorizer

    model_path = Path(config.vector.model_path).expanduser()
    return lambda: GGUFVectorizer(
        model_path=model_path,
        gpu_layers=config.vector.gpu_layers,
    )


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
        core_pipeline=True,
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

    def retrieve_context_v2(
        self, request: RetrieveContextV2Command
    ) -> RetrieveContextV2Result:
        return self._core_gateway.retrieve_context_v2(request)

    def record_retrieval_outcome_v2(
        self, command: RecordRetrievalOutcomeV2Command
    ) -> None:
        self._core_gateway.record_retrieval_outcome_v2(command)


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
    requirements = build_core_isolation_requirements(
        config.core_security,
        verify_core_signature=config.verify_core_signature,
    )
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
        self._restore_service: CoordinatedRestoreService | None = None
        self._restore_status: dict[str, object] | None = None
        self._required_core_capabilities: tuple[str, ...] = ()
        self._worker_observations: dict[str, dict[str, object]] = {}
        self._shutdown_incomplete = False
        self._shutdown_timed_out_workers: list[str] = []
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
    def restore_service(self) -> CoordinatedRestoreService | None:
        """Return the B1 coordinated restore owner for this runtime."""

        return self._restore_service

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
        if self._shutdown_incomplete:
            if any(handle.loop.is_alive() for handle in self._workers.values()):
                raise RuntimeError("Local runtime shutdown is incomplete")
            self.stop()
        if self._starting:
            raise RuntimeError("Local runtime startup is already in progress")
        self._starting = True
        self._stop_requested = False
        self._component_errors.clear()
        self._workers.clear()
        self._worker_observations.clear()
        self._core_error_code = None
        self._core_ready = False
        self._required_core_capabilities = ()
        self._restore_status = None
        self._restore_service = None
        self._shutdown_incomplete = False
        self._shutdown_timed_out_workers = []
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
            self._recover_restore_journal()
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

    def embed_query(self, memory_space_id: str, query: str) -> tuple[float, ...]:
        """Compute a retrieval embedding through the Local technical backend."""

        return self.embed_query_with_metadata(memory_space_id, query)[0]

    def embed_query_with_metadata(
        self, memory_space_id: str, query: str
    ) -> tuple[tuple[float, ...], str, str]:
        """Compute a retrieval embedding and preserve its model identity."""

        resolver = DatabaseBackedProfileResolver(self.database_path)
        profile = resolver.resolve_profile(memory_space_id, ProfileSlot.EMBEDDING)
        provider = EmbeddingProvider(
            vectorizer_factory=_build_runtime_vectorizer_factory(self.config)
        )
        batch = provider.embed((query,), profile, "retrieval_query")
        return batch.vectors[0], batch.model, batch.model_version

    def stop(self) -> None:
        """Stop workers before Core and preserve incomplete shutdown state."""

        if not self._started and self._service_lock is None and not self._pid_owned:
            return
        self.request_stop()
        timed_out_workers: list[str] = []
        for name in reversed(self._WORKER_ORDER):
            handle = self._workers.get(name)
            if handle is None:
                continue
            try:
                result = handle.loop.shutdown(handle.config.shutdown_timeout_seconds)
                stopped = bool(getattr(result, "stopped", False))
            except Exception as exc:  # noqa: BLE001 - shutdown remains observable
                stopped = False
                self._component_errors[name] = _safe_error_code(exc)
                handle.state.mark_shutdown_timed_out()
            if not stopped:
                timed_out_workers.append(name)
                self._component_errors[name] = "shutdown_timeout"

        if timed_out_workers:
            # Keep worker handles, the Core gateway, the lock and all resources
            # reachable by a live worker.  A later stop() can finish the join.
            self._shutdown_incomplete = True
            self._shutdown_timed_out_workers = list(reversed(timed_out_workers))
            self._started = False
            return

        self._shutdown_incomplete = False
        self._shutdown_timed_out_workers = []
        self._workers.clear()
        self._context_search = None
        self._backup_service = None
        self._restore_service = None
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

    def _capabilities_report(self) -> dict[str, object]:
        gateway = self.core_gateway
        advertised_operations = getattr(gateway, "advertised_operations", ())
        advertised_capabilities = getattr(gateway, "advertised_capabilities", ())

        def _safe_strings(value: object) -> list[str]:
            if not isinstance(value, (list, tuple, set, frozenset)):
                return []
            return sorted(item for item in value if isinstance(item, str))

        return {
            "ready": self._core_ready,
            "required": list(self._required_core_capabilities),
            "advertised_operations": _safe_strings(advertised_operations),
            "advertised_capabilities": _safe_strings(advertised_capabilities),
        }

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
                    "last_progress_at": state.last_progress_at,
                    "shutdown_timed_out": state.shutdown_timed_out,
                    "degraded": state.degraded,
                },
                "observability": dict(self._worker_observations.get(name, {})),
            }
            workers_ready = workers_ready and ready

        isolation = self._isolation_report()
        capabilities = self._capabilities_report()
        core_report = {
            "ready": self._core_ready,
            "available": self.core_gateway is not None,
            "error_code": self._core_error_code,
            "isolation": isolation,
            "capabilities": capabilities,
        }
        capture_ready = self.capture_ready
        core_security_ready = bool(isolation.get("ready", True))
        inference_ready = not bool(self._component_errors.get("processing"))
        if self.config.workers.processing.enabled:
            inference_ready = inference_ready and bool(self.config.hypothesis_profile_id)
        restore = dict(self._restore_status or {"ready": True, "state": "clean"})
        restore_ready = bool(restore.get("ready", False))
        capabilities_ready = bool(capabilities.get("ready", False))
        shutdown = {
            "incomplete": self._shutdown_incomplete,
            "timed_out_workers": list(self._shutdown_timed_out_workers),
        }
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
            and capabilities_ready
            and workers_ready
            and inference_ready
            and restore_ready
            and not self._shutdown_incomplete
            and not self._degraded_search
        )
        core_report["isolation"] = isolation_report
        projection_ready = bool(
            projections_report.get("ready", True)
            if isinstance(projections_report, dict)
            else True
        )
        degraded_workers = any(
            isinstance(report, dict)
            and isinstance(report.get("state"), dict)
            and bool(report["state"].get("degraded"))
            for report in worker_reports.values()
        )
        degraded = bool(
            self._degraded_search
            or degraded_workers
            or self._shutdown_incomplete
            or not restore_ready
        )
        return {
            "status": "ready" if full_ready else ("capture-ready" if capture_ready else "unavailable"),
            "capture_ready": capture_ready,
            "full_ready": full_ready,
            "degraded": degraded,
            "shutdown": shutdown,
            "components": {
                "capture": {
                    "ready": capture_ready,
                    "migrations_applied": self._migrations_applied,
                    "service_lock_held": self._lock_acquired,
                    "raw_round_writer": self._raw_round_handler is not None,
                },
                "core": core_report,
                "isolation": isolation_report,
                "capabilities": capabilities,
                "restore": restore,
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

    def _build_restore_service(self) -> CoordinatedRestoreService | None:
        backup_service = self._backup_service
        gateway = self.core_gateway
        if backup_service is None or gateway is None:
            return None
        stop_candidate = getattr(gateway, "close", None)
        health_candidate = getattr(gateway, "health", None)
        supervisor = getattr(gateway, "_supervisor", None)
        start_candidate = getattr(gateway, "start", None)
        if not callable(start_candidate):
            start_candidate = getattr(supervisor, "start", None)
        if (
            not callable(stop_candidate)
            or not callable(start_candidate)
            or not callable(health_candidate)
        ):
            raise TypeError("Core gateway cannot be restarted for restore")
        return CoordinatedRestoreService(
            core_backup_service=backup_service,
            rounds_database_path=self.database_path,
            journal_path=self.database_path.with_name("restore-journal.json"),
            stop_core=stop_candidate,
            start_core=start_candidate,
            health_check=health_candidate,
        )

    def _recover_restore_journal(self) -> None:
        """Recover a durable restore journal before advertising readiness."""

        journal_path = self.database_path.with_name("restore-journal.json")
        if not journal_path.exists():
            self._restore_status = {"ready": True, "state": "clean"}
            return
        if self._backup_service is None or self.core_gateway is None:
            self._restore_status = {
                "ready": False,
                "state": "pending",
                "error_code": "restore_pending",
            }
            self._component_errors["restore"] = "restore_pending"
            return
        try:
            self._restore_service = self._build_restore_service()
            if self._restore_service is None:
                raise RuntimeError("restore service is unavailable")
            result = self._restore_service.recover()
            self._restore_status = {
                "ready": True,
                "state": result.state if result is not None else "clean",
            }
        except CoordinatedRestoreError as exc:
            error_code = exc.code if exc.code else "restore_inconsistent"
            self._restore_status = {
                "ready": False,
                "state": "restore_inconsistent" if exc.inconsistent else "pending",
                "error_code": error_code,
            }
            self._component_errors["restore"] = error_code
        except Exception as exc:  # noqa: BLE001 - startup must expose, not hide, journals
            error_code = _safe_error_code(exc)
            self._restore_status = {
                "ready": False,
                "state": "pending",
                "error_code": error_code,
            }
            self._component_errors["restore"] = error_code

    def _start_core_if_available(self) -> None:
        try:
            if self.core_gateway is None:
                self.core_gateway = self.core_gateway_factory()
            if self.core_gateway is None:
                raise RuntimeError("Core gateway is unavailable")
            required = ["base", "maintenance", "object_facet_v2"]
            workers = self.config.workers
            if workers.core_projections.enabled:
                required.append("projections")
            if workers.core_model_tasks.enabled:
                required.append("execution_tasks")
            self._required_core_capabilities = tuple(required)
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
            secret_store = SecretStore(
                self.paths.resolve_home_path(self.config.inference_secrets_path)
            )
            profile_resolver = DatabaseBackedProfileResolver(self.database_path)
            return CoreExecutionTaskWorker(
                database_path=self.database_path,
                gateway=self.core_gateway,
                executor=CoreTaskExecutor(
                    json_provider=StructuredJsonProvider(
                        profile_resolver=profile_resolver,
                        secret_store=secret_store,
                    ),
                    embedding_provider=EmbeddingProvider(
                        vectorizer_factory=_build_runtime_vectorizer_factory(self.config)
                    ),
                    profile_resolver=profile_resolver,
                    generate_json_timeout_seconds=300,
                    embed_texts_timeout_seconds=300,
                ),
                worker_id="local-execution-tasks",
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
        observer = lambda result, observed_state: self._observe_worker_result(
            name, result, observed_state
        )
        create_loop = getattr(worker, "create_loop", None)
        if callable(create_loop):
            loop = create_loop(
                poll_interval_seconds=config_worker.interval_seconds,
                max_backoff_seconds=config_worker.max_backoff_seconds,
            )
            # A5 owns the observer hook; the A5 worker implementations that
            # predate B3 expose it on the constructed loop rather than their
            # create_loop signature.
            loop._result_observer = observer
        else:
            loop = GuardedWorkerLoop(
                worker,  # type: ignore[arg-type]
                state=state,
                name=name,
                poll_interval_seconds=config_worker.interval_seconds,
                max_backoff_seconds=config_worker.max_backoff_seconds,
                close_on_stop=callable(getattr(worker, "close", None)),
                result_observer=observer,
            )
        if not isinstance(loop, GuardedWorkerLoop):
            raise TypeError("worker factory returned an unsupported loop")
        loop.start()
        return _RuntimeWorkerHandle(name, config_worker, worker, loop, state)

    def _observe_worker_result(
        self,
        name: str,
        result: object | None,
        state: WorkerState,
    ) -> None:
        """Record B2 counters through the A5 observer without result payloads."""

        if not isinstance(result, CoreModelTaskWorkerStats):
            return
        observation: dict[str, object] = {
            "fetched": result.fetched,
            "completed": result.completed,
            "duplicates": result.duplicates,
            "failed": result.failed,
            "released": result.released,
            "retryable_failures": result.retryable_failures,
            "permanent_failures": result.permanent_failures,
            "retry_scheduled": result.retry_scheduled,
            "terminal_failures": result.terminal_failures,
            "provider_failures": result.provider_failures,
            "core_poll_failures": result.core_poll_failures,
            "core_delivery_failures": result.core_delivery_failures,
            "last_error_code": result.last_error_code,
            "degraded": result.degraded,
        }
        self._worker_observations[name] = observation
        if result.made_progress:
            state.mark_progress()
        if result.degraded:
            state.mark_degraded()

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
        missing = build_core_isolation_requirements(
            self.config.core_security,
            verify_core_signature=self.config.verify_core_signature,
        ).missing(capabilities)
        return {"ready": not missing, "missing": missing, "capabilities": payload}

    def _cleanup_after_failed_start(self) -> None:
        self.request_stop()
        timed_out_workers: list[str] = []
        for name, handle in reversed(tuple(self._workers.items())):
            try:
                result = handle.loop.shutdown(handle.config.shutdown_timeout_seconds)
                stopped = bool(getattr(result, "stopped", False))
            except Exception as exc:  # noqa: BLE001
                stopped = False
                self._component_errors[name] = _safe_error_code(exc)
                handle.state.mark_shutdown_timed_out()
            if not stopped:
                timed_out_workers.append(name)
                self._component_errors[name] = "shutdown_timeout"
        if timed_out_workers:
            self._shutdown_incomplete = True
            self._shutdown_timed_out_workers = list(reversed(timed_out_workers))
            self._started = False
            return
        self._workers.clear()
        gateway = self.core_gateway
        self.core_gateway = None
        self._backup_service = None
        self._restore_service = None
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
    cfg = config or LocalConfig(config_version=CURRENT_CONFIG_VERSION)
    if cfg.config_version < CURRENT_CONFIG_VERSION:
        cfg = cfg.model_copy(update={"config_version": CURRENT_CONFIG_VERSION})
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
        cfg = config or LocalConfig(config_version=CURRENT_CONFIG_VERSION)
        if cfg.config_version < CURRENT_CONFIG_VERSION:
            cfg = cfg.model_copy(update={"config_version": CURRENT_CONFIG_VERSION})
        _atomic_write_text(config_path, cfg.to_json(), mode=0o600)
    elif config is None:
        cfg = LocalConfig.from_file(config_path)
        # Persist the canonical schema after loading any legacy config so a
        # later startup never reintroduces the removed isolation flag.
        _atomic_write_text(config_path, cfg.to_json(), mode=0o600)
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
