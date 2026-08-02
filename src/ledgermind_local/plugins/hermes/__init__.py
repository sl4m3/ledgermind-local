"""Hermes plugin entrypoint."""

from __future__ import annotations

from typing import Any

from .hooks import PluginRuntime


def register(ctx: Any) -> None:
    runtime = PluginRuntime.build(ctx)
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_llm_call", runtime.on_pre_llm_call)
        ctx.register_hook("post_llm_call", runtime.on_post_llm_call)
        ctx.register_hook("on_session_end", runtime.on_session_end)

    runtime.start_delivery_worker()
