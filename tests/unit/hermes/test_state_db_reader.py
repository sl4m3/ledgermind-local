"""Fixtures and tests for Hermes `state.db` round reference resolution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ledgermind_local.plugins.hermes.state_db_reader import resolve_round_reference


def _create_full_state_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                api_content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp INTEGER,
                active INTEGER,
                compacted INTEGER
            )
            """,
        )
        connection.executemany(
            "INSERT INTO messages (id, session_id, role, content, api_content, tool_call_id, tool_calls, tool_name, timestamp, active, compacted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "session-1", "user", "Вступление", "Вступление", None, None, None, 1, 1, 0),
                (2, "session-1", "assistant", "Привет", "Привет", None, None, None, 2, 1, 0),
                (
                    3,
                    "session-1",
                    "user",
                    "Повтори формулу",
                    "Повтори формулу",
                    None,
                    None,
                    None,
                    3,
                    1,
                    0,
                ),
                (
                    4,
                    "session-1",
                    "assistant",
                    "Старый итог",
                    "Старый итог",
                    "call-old",
                    '[{"id":"call-old","name":"calc","arguments":"{}"}]',
                    None,
                    4,
                    1,
                    0,
                ),
                (
                    5,
                    "session-1",
                    "tool",
                    "Результат старого вызова",
                    "Результат старого вызова",
                    "call-old",
                    None,
                    "calc",
                    5,
                    1,
                    0,
                ),
                (6, "session-1", "assistant", "Финальный старый ответ", "Финальный старый ответ", None, None, None, 6, 1, 0),
                (
                    7,
                    "session-1",
                    "user",
                    "Повтори формулу",
                    "Повтори формулу",
                    None,
                    None,
                    None,
                    7,
                    1,
                    0,
                ),
                (
                    8,
                    "session-1",
                    "assistant",
                    "Новый промежуточный ответ",
                    "Новый промежуточный ответ",
                    "call-new",
                    '[{"id":"call-new","name":"calc","arguments":"{\"x\":1}"}]',
                    None,
                    8,
                    1,
                    0,
                ),
                (
                    9,
                    "session-1",
                    "tool",
                    "Результат нового вызова",
                    "Результат нового вызова",
                    "call-new",
                    None,
                    "calc",
                    9,
                    1,
                    0,
                ),
                (10, "session-1", "assistant", "Финал нового ответа", "Финал нового ответа", None, None, None, 10, 1, 0),
                (
                    11,
                    "session-1",
                    "user",
                    "Архивный пользователь",
                    "Архивный пользователь",
                    None,
                    None,
                    None,
                    11,
                    0,
                    1,
                ),
                (
                    12,
                    "session-1",
                    "assistant",
                    "Архивный ответ",
                    "Архивный ответ",
                    None,
                    None,
                    None,
                    12,
                    0,
                    1,
                ),
            ],
        )
        connection.commit()


def _create_minimal_state_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """,
        )
        connection.executemany(
            "INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
            [
                (1, "session-1", "user", "первый запрос"),
                (2, "session-1", "assistant", "первый ответ"),
                (3, "session-1", "user", "повторный запрос"),
                (4, "session-1", "assistant", "ответ №1"),
                (5, "session-1", "user", "повторный запрос"),
                (6, "session-1", "assistant", "ответ №2"),
            ],
        )
        connection.commit()


def test_resolve_round_reference_prefers_latest_duplicate_round(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_full_state_db(state_db)

    reference = resolve_round_reference(
        session_id="session-1",
        user_message="Повтори формулу",
        assistant_response="Финал нового ответа",
        state_db_path=state_db,
        source_schema_version=1,
        resolver_version=1,
    )

    assert reference.source_session_id == "session-1"
    assert reference.first_message_id == "7"
    assert reference.final_message_id == "10"
    assert reference.message_ids == ("7", "8", "9", "10")
    assert reference.source_round_id == "7:10"


def test_resolve_round_reference_uses_checkpoint_to_restrict_search(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_full_state_db(state_db)

    reference = resolve_round_reference(
        session_id="session-1",
        user_message="Повтори формулу",
        assistant_response="Финал нового ответа",
        state_db_path=state_db,
        last_completed_message_id="4",
        source_schema_version=1,
        resolver_version=1,
    )

    assert reference.first_message_id == "7"
    assert reference.final_message_id == "10"
    assert reference.message_ids == ("7", "8", "9", "10")


def test_resolve_round_reference_works_without_optional_columns(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_minimal_state_db(state_db)

    reference = resolve_round_reference(
        session_id="session-1",
        user_message="повторный запрос",
        assistant_response="ответ №2",
        state_db_path=state_db,
        source_schema_version=1,
        resolver_version=1,
    )

    assert reference.source_session_id == "session-1"
    assert reference.first_message_id == "5"
    assert reference.final_message_id == "6"
    assert reference.message_ids == ("5", "6")
    assert reference.source_round_id == "5:6"


def test_resolve_round_reference_prefers_turn_id_over_derived_round_id(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_full_state_db(state_db)

    reference = resolve_round_reference(
        session_id="session-1",
        user_message="Повтори формулу",
        assistant_response="Финал нового ответа",
        turn_id="turn-123",
        state_db_path=state_db,
        source_schema_version=1,
        resolver_version=1,
    )

    assert reference.source_round_id == "turn-123"
    assert reference.first_message_id == "7"
    assert reference.final_message_id == "10"


def test_resolve_round_reference_uses_api_request_id_when_turn_id_absent(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_minimal_state_db(state_db)

    reference = resolve_round_reference(
        session_id="session-1",
        user_message="повторный запрос",
        assistant_response="ответ №2",
        api_request_id="api-999",
        state_db_path=state_db,
        source_schema_version=1,
        resolver_version=1,
    )

    assert reference.source_round_id == "api-999"
    assert reference.first_message_id == "5"
    assert reference.final_message_id == "6"
