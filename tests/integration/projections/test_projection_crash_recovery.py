"""Integration tests for crash recovery of derived projections."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC_ROOT = REPO_ROOT / "src"
CORE_SRC_ROOT = REPO_ROOT.parent / "ledgermind-core" / "src"
for extra_root in (CORE_SRC_ROOT, PROJECT_SRC_ROOT):
    if extra_root.exists() and str(extra_root) not in sys.path:
        sys.path.insert(0, str(extra_root))

from persistence import open_sqlite_connection
from projections import VectorProjectionStore as ProjectionVectorStore


PYTHON_PATHS = [str(path) for path in (CORE_SRC_ROOT, PROJECT_SRC_ROOT) if path.exists()]


def _runner_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [*PYTHON_PATHS, *(os.getenv("PYTHONPATH").split(os.pathsep) if os.getenv("PYTHONPATH") else [])]
        ),
    }


def _write_recovery_script(tmp_path: Path) -> Path:
    script = tmp_path / "projection_recovery_runner.py"
    script_body = textwrap.dedent(
        '''
        from __future__ import annotations

        import argparse
        from datetime import datetime, timedelta, timezone
        import json
        import os
        import signal
        from pathlib import Path
        from typing import Sequence

        from persistence import (
            OutboxEvent,
            SQLiteOutboxRepository,
            SQLiteUnitOfWork,
            migrations,
            open_sqlite_connection,
        )
        from projections import (
            ProjectionDispatcher,
            KnowledgeFTSProjection,
            KnowledgeMarkdownGitAuditProjection,
            KnowledgeMarkdownProjection,
            KnowledgeVectorProjection,
        )
        from projections.vector_store import VectorProjectionStore

        _NOW = "2026-01-01T00:00:00+00:00"

        _RETRY_DELAYS_SECONDS = (1, 5, 30, 300, 1800)


        def _parse_projections(raw: str) -> tuple[str, ...]:
            return tuple(
                value.strip()
                for value in (raw or "").split(",")
                if value.strip()
            )


        def _bootstrap(database: Path) -> None:
            connection = open_sqlite_connection(database)
            try:
                migrations.apply_migrations(connection)
                connection.commit()
            finally:
                connection.close()


        def _seed_projection_event(
            database: Path,
            *,
            event_id: str,
            memory_space_id: str,
            knowledge_id: str,
            projection_names: tuple[str, ...],
            title: str,
            target: str,
            statement: str,
        ) -> None:
            connection = open_sqlite_connection(database)
            try:
                migrations.apply_migrations(connection)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_spaces (
                        memory_space_id,
                        display_name,
                        source_client,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_space_id, None, "hermes", _NOW, _NOW),
                )
                connection.execute(
                    """
                    INSERT INTO knowledge_items (
                        knowledge_id,
                        memory_space_id,
                        title,
                        target,
                        statement,
                        rationale,
                        phase,
                        version,
                        created_at,
                        updated_at,
                        superseded_by_id,
                        deleted_at
                    ) VALUES (?, ?, ?, ?, ?, '', 'pattern', 1, ?, ?, NULL, NULL)
                    ON CONFLICT(knowledge_id) DO UPDATE SET
                        memory_space_id = excluded.memory_space_id,
                        title = excluded.title,
                        target = excluded.target,
                        statement = excluded.statement,
                        rationale = excluded.rationale,
                        updated_at = excluded.updated_at
                    """,
                    (knowledge_id, memory_space_id, title, target, statement, _NOW, _NOW),
                )

                if projection_names:
                    outbox = SQLiteOutboxRepository(connection)
                    payload = json.dumps(
                        {
                            "event_type": "knowledge.created",
                            "knowledge_id": knowledge_id,
                        }
                    )
                    outbox.add(
                        OutboxEvent(
                            event_id=event_id,
                            event_type="knowledge.created",
                            aggregate_id=knowledge_id,
                            memory_space_id=memory_space_id,
                            payload_json=payload,
                            occurred_at=_NOW,
                            available_at=_NOW,
                            attempts=0,
                            claimed_at=None,
                            claimed_by=None,
                            processed_at=None,
                            last_error=None,
                        ),
                        projection_names=projection_names,
                    )

                connection.commit()
            finally:
                connection.close()


        class _CrashAfterMarkdownProjection:
            def __init__(self, projection: KnowledgeMarkdownProjection, marker_path: Path) -> None:
                self._projection = projection
                self._marker_path = marker_path

            def handle_event(self, *, event_type: str, memory_space_id: str, aggregate_id: str, payload_json: str) -> bool:
                changed = self._projection.handle_event(
                    event_type=event_type,
                    memory_space_id=memory_space_id,
                    aggregate_id=aggregate_id,
                    payload_json=payload_json,
                )
                if not self._marker_path.exists():
                    self._marker_path.write_text("crashed", encoding="utf-8")
                    os.kill(os.getpid(), signal.SIGKILL)
                return changed


        class _RecoveryVectorizer:
            @property
            def fingerprint(self) -> str:
                return "recovery-vectorizer"

            @property
            def dimension(self) -> int:
                return 2

            @property
            def model_name(self) -> str:
                return "recovery-vectorizer"

            def encode(self, texts: list[str]) -> list[tuple[float, float]]:
                return [(float(len(text)), float(len(texts))) for text in texts]

            def close(self) -> None:
                return None


        def _run_vector_rebuild(database: Path, *, crash_on_write: bool) -> None:
            connection = open_sqlite_connection(database)
            try:
                migrations.apply_migrations(connection)
                vector_root = database.with_suffix(".vectors")
                projection = KnowledgeVectorProjection(
                    connection=connection,
                    vector_store_root=vector_root,
                    vectorizer_factory=_RecoveryVectorizer,
                )

                if crash_on_write:
                    original = VectorProjectionStore._write_vectors

                    def _write_vectors(self, path: Path, vectors: list[list[float]]) -> None:
                        original(self, path, vectors)
                        os._exit(1)

                    VectorProjectionStore._write_vectors = _write_vectors

                projection.rebuild()
                projection.close()
                connection.commit()
            finally:
                connection.close()


        class _SimpleVectorProjectionFactory:
            def __call__(self) -> _RecoveryVectorizer:
                return _RecoveryVectorizer()


        def _build_handlers(
            connection,
            database: Path,
            projections: Sequence[str],
            markdown_root: Path,
            crash_after_markdown: bool,
            marker_path: Path | None,
        ) -> dict[str, object]:
            handlers: dict[str, object] = {}
            for projection_name in projections:
                if projection_name == "projections.search":
                    handlers[projection_name] = KnowledgeFTSProjection(connection=connection)
                elif projection_name == "projections.knowledge":
                    handlers[projection_name] = KnowledgeVectorProjection(
                        connection=connection,
                        vector_store_root=database.with_suffix(".vectors"),
                        vectorizer_factory=_SimpleVectorProjectionFactory(),
                    )
                elif projection_name == "projections.markdown":
                    projection = KnowledgeMarkdownProjection(
                        connection=connection,
                        markdown_root=markdown_root,
                    )
                    if crash_after_markdown and marker_path is not None:
                        projection = _CrashAfterMarkdownProjection(projection, marker_path)
                    handlers[projection_name] = projection
                elif projection_name == "projections.markdown_audit":
                    handlers[projection_name] = KnowledgeMarkdownGitAuditProjection(
                        markdown_root=markdown_root,
                        enabled=True,
                        batch_size=1,
                    )
            return handlers


        def _to_iso(value: datetime) -> str:
            if value.tzinfo is None:
                raise ValueError("timestamp must be timezone-aware")
            return value.isoformat()


        def _retry_delay_seconds(attempt: int) -> int:
            if attempt <= 1:
                return _RETRY_DELAYS_SECONDS[0]
            if attempt >= len(_RETRY_DELAYS_SECONDS):
                return _RETRY_DELAYS_SECONDS[-1]
            return _RETRY_DELAYS_SECONDS[attempt - 1]


        def _run_worker(
            database: Path,
            projections: tuple[str, ...],
            markdown_root: Path | None,
            crash_after_markdown: bool,
            marker_path: Path | None,
        ) -> None:
            resolved_markdown_root = markdown_root or Path(str(database)).with_suffix(".markdown")
            for projection_name in projections:
                while True:
                    with SQLiteUnitOfWork(database) as uow:
                        connection = uow.connection
                        migrations.apply_migrations(connection)
                        handlers = _build_handlers(
                            connection=connection,
                            database=database,
                            projections=(projection_name,),
                            markdown_root=resolved_markdown_root,
                            crash_after_markdown=crash_after_markdown,
                            marker_path=marker_path,
                        )
                        dispatcher = ProjectionDispatcher(handlers)

                        now = datetime.now(timezone.utc)
                        stale_cutoff = now - timedelta(seconds=30)
                        event = uow.outbox.acquire_next(
                            projection_name=projection_name,
                            worker_id="crash-recovery-worker",
                            now=_to_iso(now),
                            stale_claim_before=_to_iso(stale_cutoff),
                        )
                        if event is None:
                            break

                        try:
                            dispatcher.dispatch(projection_name, event)
                        except Exception as exc:  # noqa: BLE001
                            uow.outbox.mark_failed(
                                projection_name,
                                event.event_id,
                                available_at=_to_iso(
                                    now + timedelta(seconds=_retry_delay_seconds(event.attempts + 1))
                                ),
                                last_error=f"{type(exc).__name__}: {exc}",
                            )
                        else:
                            uow.outbox.mark_processed(
                                projection_name,
                                event.event_id,
                                processed_at=_to_iso(now),
                            )
                        uow.commit()


        def _main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "command",
                choices=("seed", "run-worker", "run-vector-rebuild"),
            )
            parser.add_argument("database")
            parser.add_argument("--event-id", default="evt-1")
            parser.add_argument("--memory-space-id", default="space-a")
            parser.add_argument("--knowledge-id", default="k-1")
            parser.add_argument("--title", default="Recovered title")
            parser.add_argument("--target", default="Recovered target")
            parser.add_argument("--statement", default="Recovered statement")
            parser.add_argument("--projections", default="")
            parser.add_argument("--kill-after-seed", action="store_true")
            parser.add_argument("--worker-id", default="crash-recovery-worker")
            parser.add_argument("--markdown-root")
            parser.add_argument("--crash-after-markdown", action="store_true")
            parser.add_argument("--crash-marker")
            parser.add_argument("--crash-on-vector-write", action="store_true")

            args = parser.parse_args()
            database = Path(args.database)
            projections = _parse_projections(args.projections)
            command = args.command

            if command == "seed":
                _seed_projection_event(
                    database,
                    event_id=args.event_id,
                    memory_space_id=args.memory_space_id,
                    knowledge_id=args.knowledge_id,
                    projection_names=projections,
                    title=args.title,
                    target=args.target,
                    statement=args.statement,
                )
                if args.kill_after_seed:
                    os.kill(os.getpid(), signal.SIGKILL)
                return 0

            if command == "run-worker":
                _run_worker(
                    database=database,
                    projections=projections,
                    markdown_root=Path(args.markdown_root) if args.markdown_root else None,
                    crash_after_markdown=args.crash_after_markdown,
                    marker_path=Path(args.crash_marker) if args.crash_marker else None,
                )
                return 0

            if command == "run-vector-rebuild":
                _run_vector_rebuild(database, crash_on_write=args.crash_on_vector_write)
                return 0

            return 2


        if __name__ == "__main__":
            raise SystemExit(_main())
        '''
    )
    script.write_text(
        (
            script_body.strip()
            + "\n"
        ),
        encoding="utf-8",
    )
    return script


def _run_script(script: Path, *arguments: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        env=_runner_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run_sql_scalar(database: Path, query: str, args: tuple[object, ...] = ()) -> object | None:
    conn = open_sqlite_connection(database)
    try:
        row = conn.execute(query, args).fetchone()
        if row is None:
            return None
        return row[0] if len(row.keys()) == 1 else row
    finally:
        conn.close()


def _git_commit_count(markdown_root: Path) -> int:
    command = subprocess.run(
        ["git", "-C", str(markdown_root), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if command.returncode != 0:
        return 0
    return int((command.stdout or "0").strip() or 0)


def test_projection_recovery_restores_search_projection_after_process_kill(tmp_path: Path) -> None:
    script = _write_recovery_script(tmp_path)
    db = tmp_path / "state.db"

    seed = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-fts-recover",
        "--projections",
        "projections.search",
        "--kill-after-seed",
    )
    assert seed.returncode == -signal.SIGKILL
    assert _run_sql_scalar(db, "SELECT processed_at FROM outbox_events WHERE event_id = ?", ("evt-fts-recover",)) is None
    assert (
        _run_sql_scalar(
            db,
            "SELECT COUNT(*) AS total FROM projection_deliveries WHERE event_id = ? AND projection_name = 'projections.search'",
            ("evt-fts-recover",),
        )
        == 1
    )

    rebuilt = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.search",
    )
    assert rebuilt.returncode == 0

    assert (
        _run_sql_scalar(
            db,
            'SELECT COUNT(*) AS total FROM knowledge_fts WHERE knowledge_id = "k-1"'
        )
        == 1
    )
    assert (
        _run_sql_scalar(
            db,
            "SELECT processed_at FROM outbox_events WHERE event_id = ?",
            ("evt-fts-recover",),
        )
        is not None
    )


def test_vector_rebuild_preserves_previous_index_after_mid_wirte_crash(tmp_path: Path) -> None:
    script = _write_recovery_script(tmp_path)
    db = tmp_path / "state.db"
    projections = "projections.knowledge"

    seed = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-vector-seed",
        "--projections",
        projections,
        "--knowledge-id",
        "k-vector",
        "--title",
        "Vector title",
    )
    assert seed.returncode == 0

    warmup = _run_script(
        script,
        "run-vector-rebuild",
        str(db),
    )
    assert warmup.returncode == 0

    vector_root = db.with_suffix(".vectors")
    baseline = ProjectionVectorStore(vector_root)
    assert baseline.ids == ("k-vector",)

    crashed = _run_script(
        script,
        "run-vector-rebuild",
        str(db),
        "--crash-on-vector-write",
        timeout=20.0,
    )
    assert crashed.returncode == 1

    current = ProjectionVectorStore(vector_root)
    assert current.ids == baseline.ids
    assert current.document_count == baseline.document_count


def test_markdown_and_markdown_audit_recover_when_process_kills_after_markdown(tmp_path: Path) -> None:
    script = _write_recovery_script(tmp_path)
    db = tmp_path / "state.db"
    markdown_root = tmp_path / "markdown"
    marker = tmp_path / "markdown.crash.marker"

    seed = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-md-audit",
        "--knowledge-id",
        "k-md",
        "--title",
        "Recovered markdown",
        "--target",
        "Recovered target",
        "--statement",
        "Recovered statement",
        "--projections",
        "projections.markdown,projections.markdown_audit",
    )
    assert seed.returncode == 0

    crashed = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.markdown,projections.markdown_audit",
        "--markdown-root",
        str(markdown_root),
        "--crash-after-markdown",
        "--crash-marker",
        str(marker),
    )
    assert crashed.returncode == -signal.SIGKILL
    assert _git_commit_count(markdown_root) == 0

    recovered = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.markdown,projections.markdown_audit",
        "--markdown-root",
        str(markdown_root),
    )
    assert recovered.returncode == 0
    assert _git_commit_count(markdown_root) == 1
    assert len(list((markdown_root / "knowledge").rglob("*.md"))) == 1

    replay = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-md-audit-replay",
        "--knowledge-id",
        "k-md",
        "--title",
        "Recovered markdown",
        "--target",
        "Recovered target",
        "--statement",
        "Recovered statement",
        "--projections",
        "projections.markdown,projections.markdown_audit",
    )
    assert replay.returncode == 0

    replayed = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.markdown,projections.markdown_audit",
        "--markdown-root",
        str(markdown_root),
    )
    assert replayed.returncode == 0
    assert _git_commit_count(markdown_root) == 1
    assert len(list((markdown_root / "knowledge").rglob("*.md"))) == 1


def test_replayed_outbox_event_does_not_duplicate_vector_records(tmp_path: Path) -> None:
    script = _write_recovery_script(tmp_path)
    db = tmp_path / "state.db"
    vector_root = db.with_suffix(".vectors")

    first = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-vector-replay-1",
        "--knowledge-id",
        "k-repeat",
        "--projections",
        "projections.knowledge",
    )
    assert first.returncode == 0

    process_first = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.knowledge",
    )
    assert process_first.returncode == 0
    assert ProjectionVectorStore(vector_root).ids == ("k-repeat",)

    second = _run_script(
        script,
        "seed",
        str(db),
        "--event-id",
        "evt-vector-replay-2",
        "--knowledge-id",
        "k-repeat",
        "--projections",
        "projections.knowledge",
    )
    assert second.returncode == 0

    process_second = _run_script(
        script,
        "run-worker",
        str(db),
        "--projections",
        "projections.knowledge",
    )
    assert process_second.returncode == 0

    final_ids = ProjectionVectorStore(vector_root).ids
    assert final_ids == ("k-repeat",)
