"""Small logging helpers for the Hermes plugin."""

from __future__ import annotations

import uuid


def new_error_id() -> str:
    return f"err_{uuid.uuid4().hex}"


def sanitize_message_for_log(message: str, *, max_length: int = 1_000) -> str:
    if len(message) <= max_length:
        return message
    return f"{message[:max_length]}…"
