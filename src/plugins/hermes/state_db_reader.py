"""Read-only resolver for Hermes `state.db` round boundaries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResolvedRoundReference:
    source_session_id: str | None
    source_round_id: str
    first_message_id: str | None
    final_message_id: str | None
    message_ids: tuple[str, ...]
    source_digest: str | None = None
    source_schema_version: int = 1
    resolver_version: int = 1


class StateDBAccessError(RuntimeError):
    """State database could not be opened or queried."""


class StateDBSchemaError(RuntimeError):
    """Unsupported Hermes schema version for `state.db`."""


_REQUIRED_MESSAGE_COLUMNS = {
    "id",
    "session_id",
    "role",
    "content",
}

_MESSAGE_COLUMN_ALIASES = {
    "role": "role",
    "session_id": "session_id",
    "id": "id",
    "content": "content",
    "api_content": "api_content",
    "tool_call_id": "tool_call_id",
    "tool_calls": "tool_calls",
    "tool_name": "tool_name",
    "timestamp": "timestamp",
    "active": "active",
    "compacted": "compacted",
}

_DEFAULT_FALLBACK_MESSAGE_WINDOW = 120


def _to_str(value: object | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _read_state_db_message_columns(db_path: Path) -> set[str]:
    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        timeout=1.0,
        uri=True,
        check_same_thread=False,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 1000")

        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'",
        ).fetchone()
        if row is None:
            raise StateDBSchemaError("missing messages table")

        columns = {
            str(item["name"]) for item in connection.execute("PRAGMA table_info(messages)")
        }
        missing = _REQUIRED_MESSAGE_COLUMNS - columns
        if missing:
            raise StateDBSchemaError(
                f"messages table missing required columns: {', '.join(sorted(missing))}",
            )
        return columns
    finally:
        connection.close()


def _connect_state_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser()
    if not db_path.exists():
        raise StateDBAccessError("state.db is missing")

    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        timeout=1.0,
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 1000")
    return connection


def _message_body(row: sqlite3.Row, columns: set[str]) -> str:
    if "api_content" in columns:
        api_content = row["api_content"]
        if _to_str(api_content) is not None:
            return str(api_content)
    return str(row["content"] or "")


def _safe_json_parse(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool, type(None))):
        return value
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _fetch_messages(
    connection: sqlite3.Connection,
    session_id: str,
    columns: set[str],
    *,
    after_message_id: int | None = None,
    limit: int | None = None,
    order_desc: bool = False,
) -> list[sqlite3.Row]:
    selected = [
        _MESSAGE_COLUMN_ALIASES[column]
        for column in (
            "id",
            "session_id",
            "role",
            "content",
            "api_content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "timestamp",
            "active",
            "compacted",
        )
        if column in columns
    ]

    query = f"SELECT {', '.join(selected)} FROM messages WHERE session_id = ?"
    params: list[object] = [session_id]
    if after_message_id is not None:
        query += " AND id > ?"
        params.append(after_message_id)

    query += " ORDER BY id DESC" if order_desc else " ORDER BY id ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    return list(connection.execute(query, params).fetchall())


def _find_assistant_index(
    messages: list[sqlite3.Row],
    columns: set[str],
    assistant_response: str,
) -> int:
    if not messages:
        return -1

    expected = str(assistant_response or "").strip()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message["role"]).strip().lower() != "assistant":
            continue

        content = _message_body(message, columns).strip()
        if expected and content == expected:
            return index

    if expected:
        return -1

    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index]["role"]).strip().lower() == "assistant":
            return index
    return -1


def _find_user_index(
    messages: list[sqlite3.Row],
    columns: set[str],
    assistant_index: int,
    user_message: str,
) -> int:
    if assistant_index <= 0:
        return max(assistant_index - 1, 0)

    expected = str(user_message or "").strip()
    for index in range(assistant_index - 1, -1, -1):
        if str(messages[index]["role"]).strip().lower() != "user":
            continue
        content = _message_body(messages[index], columns).strip()
        if not expected or content == expected:
            return index

    for index in range(assistant_index - 1, -1, -1):
        if str(messages[index]["role"]).strip().lower() == "user":
            return index

    return max(assistant_index - 1, 0)


def _compute_round_messages(
    messages: list[sqlite3.Row],
    columns: set[str],
    user_message: str,
    assistant_response: str,
) -> list[sqlite3.Row]:
    assistant_index = _find_assistant_index(
        messages=messages,
        columns=columns,
        assistant_response=assistant_response,
    )
    if assistant_index < 0:
        return []

    user_index = _find_user_index(
        messages=messages,
        columns=columns,
        assistant_index=assistant_index,
        user_message=user_message,
    )
    return messages[user_index : assistant_index + 1]


def _compute_digest(
    *,
    session_id: str | None,
    source_round_id: str,
    message_ids: tuple[str, ...],
    user_message: str,
    assistant_response: str,
) -> str:
    payload = {
        "session_id": session_id or "",
        "source_round_id": source_round_id,
        "message_ids": list(message_ids),
        "user_message": user_message or "",
        "assistant_response": assistant_response or "",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _build_round_id(
    *,
    explicit_round_id: str | None,
    first_message_id: str | None,
    final_message_id: str | None,
    session_id: str | None,
    user_message: str,
    assistant_response: str,
    message_ids: tuple[str, ...],
) -> str:
    if explicit_round_id:
        return explicit_round_id
    if first_message_id and final_message_id:
        return f"{first_message_id}:{final_message_id}"
    if message_ids:
        return f"{session_id or 'session'}:{message_ids[0]}:{message_ids[-1]}"
    fallback = json.dumps(
        {
            "session_id": session_id or "",
            "user_message": user_message or "",
            "assistant_response": assistant_response or "",
            "message_count": len(message_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"fallback:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"


def _resolve_from_messages(
    columns: set[str],
    messages: list[sqlite3.Row],
    user_message: str,
    assistant_response: str,
    *,
    explicit_round_id: str | None = None,
    session_id: str | None = None,
    source_schema_version: int = 1,
    resolver_version: int = 1,
) -> ResolvedRoundReference | None:
    if not messages:
        return None

    selected = _compute_round_messages(
        messages=messages,
        columns=columns,
        user_message=user_message,
        assistant_response=assistant_response,
    )
    if not selected:
        return None

    first_id = str(selected[0]["id"])
    final_id = str(selected[-1]["id"])
    ids = tuple(str(item["id"]) for item in selected)
    round_id = _build_round_id(
        explicit_round_id=explicit_round_id,
        first_message_id=first_id,
        final_message_id=final_id,
        session_id=session_id,
        user_message=user_message,
        assistant_response=assistant_response,
        message_ids=ids,
    )
    return ResolvedRoundReference(
        source_session_id=session_id,
        source_round_id=round_id,
        first_message_id=first_id,
        final_message_id=final_id,
        message_ids=ids,
        source_digest=_compute_digest(
            session_id=session_id,
            source_round_id=round_id,
            message_ids=ids,
            user_message=user_message,
            assistant_response=assistant_response,
        ),
        source_schema_version=source_schema_version,
        resolver_version=resolver_version,
    )


def _resolve_with_checkpoint(
    connection: sqlite3.Connection,
    columns: set[str],
    *,
    session_id: str,
    user_message: str,
    assistant_response: str,
    checkpoint_message_id: str | None,
    explicit_round_id: str | None,
    source_schema_version: int,
    resolver_version: int,
) -> ResolvedRoundReference | None:
    if not checkpoint_message_id:
        return None
    try:
        after = int(checkpoint_message_id)
    except ValueError:
        return None

    messages = _fetch_messages(
        connection=connection,
        session_id=session_id,
        columns=columns,
        after_message_id=after,
        order_desc=False,
    )
    if not messages:
        return None
    return _resolve_from_messages(
        columns=columns,
        messages=messages,
        user_message=user_message,
        assistant_response=assistant_response,
        explicit_round_id=explicit_round_id,
        session_id=session_id,
        source_schema_version=source_schema_version,
        resolver_version=resolver_version,
    )


def _resolve_recent_search(
    connection: sqlite3.Connection,
    columns: set[str],
    *,
    session_id: str,
    user_message: str,
    assistant_response: str,
    explicit_round_id: str | None,
    source_schema_version: int,
    resolver_version: int,
) -> ResolvedRoundReference | None:
    recent = _fetch_messages(
        connection=connection,
        session_id=session_id,
        columns=columns,
        order_desc=True,
        limit=_DEFAULT_FALLBACK_MESSAGE_WINDOW,
    )
    if not recent:
        return None
    # Convert newest-window to chronological order.
    recent = list(recent)[::-1]
    return _resolve_from_messages(
        columns=columns,
        messages=recent,
        user_message=user_message,
        assistant_response=assistant_response,
        explicit_round_id=explicit_round_id,
        session_id=session_id,
        source_schema_version=source_schema_version,
        resolver_version=resolver_version,
    )


def _resolve_by_explicit_ids(
    *,
    session_id: str,
    user_message: str,
    assistant_response: str,
    first_message_id: str | None,
    final_message_id: str | None,
    explicit_message_ids: tuple[str, ...],
    explicit_round_id: str | None,
    source_schema_version: int,
    resolver_version: int,
) -> ResolvedRoundReference:
    ids = list(explicit_message_ids)
    if first_message_id and final_message_id:
        ids = [first_message_id, final_message_id]
    elif not ids and first_message_id:
        ids = [first_message_id]
    elif not ids and final_message_id:
        ids = [final_message_id]

    ids_tuple = tuple(ids)
    round_id = _build_round_id(
        explicit_round_id=explicit_round_id,
        first_message_id=first_message_id,
        final_message_id=final_message_id,
        session_id=session_id,
        user_message=user_message,
        assistant_response=assistant_response,
        message_ids=ids_tuple,
    )
    return ResolvedRoundReference(
        source_session_id=session_id,
        source_round_id=round_id,
        first_message_id=first_message_id,
        final_message_id=final_message_id,
        message_ids=ids_tuple,
        source_digest=_compute_digest(
            session_id=session_id,
            source_round_id=round_id,
            message_ids=ids_tuple,
            user_message=user_message,
            assistant_response=assistant_response,
        ),
        source_schema_version=source_schema_version,
        resolver_version=resolver_version,
    )


def _to_string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_to_str(value),) if _to_str(value) else ()
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(
        item
        for raw in value
        if (item := _to_str(raw)) is not None
    )


def resolve_round_reference(
    *,
    session_id: str | None,
    user_message: str,
    assistant_response: str,
    turn_id: str | None = None,
    api_request_id: str | None = None,
    state_db_path: str | Path | None = None,
    last_completed_message_id: str | None = None,
    explicit_first_message_id: str | None = None,
    explicit_final_message_id: str | None = None,
    explicit_message_ids: tuple[str, ...] | list[str] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    source_schema_version: int = 1,
    resolver_version: int = 1,
) -> ResolvedRoundReference:
    del conversation_history

    explicit_first = _to_str(explicit_first_message_id)
    explicit_final = _to_str(explicit_final_message_id)
    explicit_ids = _to_string_list(explicit_message_ids)

    preferred_round_id = _to_str(turn_id) or _to_str(api_request_id)
    if explicit_first or explicit_final or explicit_ids:
        return _resolve_by_explicit_ids(
            session_id=session_id or "",
            user_message=user_message,
            assistant_response=assistant_response,
            first_message_id=explicit_first,
            final_message_id=explicit_final,
            explicit_message_ids=explicit_ids,
            explicit_round_id=preferred_round_id,
            source_schema_version=source_schema_version,
            resolver_version=resolver_version,
        )

    if state_db_path is not None and _to_str(session_id):
        try:
            db_path = Path(state_db_path).expanduser()
            columns = _read_state_db_message_columns(db_path)
            connection = _connect_state_db(db_path)
            try:
                candidate = _resolve_with_checkpoint(
                    connection=connection,
                    columns=columns,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    checkpoint_message_id=last_completed_message_id,
                    explicit_round_id=preferred_round_id,
                    source_schema_version=source_schema_version,
                    resolver_version=resolver_version,
                )
                if candidate is not None:
                    return candidate

                candidate = _resolve_recent_search(
                    connection=connection,
                    columns=columns,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    explicit_round_id=preferred_round_id,
                    source_schema_version=source_schema_version,
                    resolver_version=resolver_version,
                )
                if candidate is not None:
                    return candidate
            finally:
                connection.close()
        except (StateDBAccessError, StateDBSchemaError, sqlite3.Error, ValueError):
            pass

    fallback_message_ids: tuple[str, ...] = ()
    source_round_id = preferred_round_id or _build_round_id(
        explicit_round_id=None,
        first_message_id=None,
        final_message_id=None,
        session_id=session_id,
        user_message=user_message,
        assistant_response=assistant_response,
        message_ids=fallback_message_ids,
    )
    source_digest = _compute_digest(
        session_id=session_id,
        source_round_id=source_round_id,
        message_ids=fallback_message_ids,
        user_message=user_message,
        assistant_response=assistant_response,
    )
    return ResolvedRoundReference(
        source_session_id=session_id,
        source_round_id=source_round_id,
        first_message_id=None,
        final_message_id=None,
        message_ids=fallback_message_ids,
        source_digest=source_digest,
        source_schema_version=source_schema_version,
        resolver_version=resolver_version,
    )
