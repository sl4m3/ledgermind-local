from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import cast

import uvicorn

from ledgermind_local.api.app import create_app
from ledgermind_local.api.dependencies import Application, Settings
from ledgermind_local.persistence import migrations, open_sqlite_connection
from ledgermind_local.plugins.hermes.client import LedgerMindClient
from ledgermind_local.plugins.hermes.delivery_worker import DeliveryWorker
from ledgermind_local.plugins.hermes.spool import FileSpool


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bootstrap_database(path: Path) -> None:
    connection = open_sqlite_connection(path)
    try:
        migrations.apply_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def _request_payload() -> dict[str, object]:
    return {
        "api_version": "1",
        "idempotency_key": _digest("hermes-e2e"),
        "memory_space_id": "e2e-space",
        "source": {
            "source_system": "hermes",
            "source_instance_id": "e2e-instance",
            "source_profile_id": "e2e-profile",
            "source_session_id": "e2e-session",
            "source_round_id": "e2e-round",
            "first_message_id": "e2e-message-1",
            "final_message_id": "e2e-message-2",
            "message_ids": ["e2e-message-1", "e2e-message-2"],
            "source_digest": _digest("hermes-round"),
            "source_schema_version": 1,
            "resolver_version": 1,
        },
        "extraction": {
            "host": "hermes",
            "provider": "test",
            "model": "test-model",
            "prompt_version": 1,
            "schema_version": 1,
            "purpose": "ledgermind.atom.extract",
        },
        "atom": {
            "title": "E2E delivery",
            "target": "tests.delivery",
            "statement": "The Hermes delivery path writes to canonical SQLite.",
            "rationale": "Exercise the real local HTTP boundary.",
            "result": "stored",
            "artifacts": [],
        },
    }


def test_hermes_file_spool_delivery_reaches_fastapi_and_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    _bootstrap_database(database)
    token_file = tmp_path / "server.token"
    token_file.write_text("e2e-token\n", encoding="utf-8")

    app = create_app(
        application=cast(Application, object()),
        settings=Settings(database_path=database, api_token="e2e-token"),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="critical",
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started is True
    assert server.servers
    port = int(server.servers[0].sockets[0].getsockname()[1])

    spool = FileSpool(tmp_path / "spool")
    request = _request_payload()
    item = spool.enqueue_ready(str(request["idempotency_key"]), request)
    client = LedgerMindClient(
        service_url=f"http://127.0.0.1:{port}",
        token_file=str(token_file),
        timeout=2.0,
    )
    worker = DeliveryWorker(spool, client, batch_size=1, request_timeout=2.0)
    try:
        assert worker._drain_once() is True
        assert not item.exists()
        assert not list(spool.ready_dir.glob("*.json"))
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM idempotency_results").fetchone()[0] == 1
