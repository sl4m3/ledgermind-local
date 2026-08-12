"""Verification of the installed Hermes payload and hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...errors import AdapterError

REQUIRED_HOOKS = {
    "pre_llm_call",
    "pre_tool_call",
    "post_tool_call",
    "post_llm_call",
    "on_session_end",
}


def verify_hermes_plugin(plugin_root: str | Path) -> dict[str, Any]:
    root = Path(plugin_root)
    manifest_path = root / "plugin.yaml"
    if not manifest_path.is_file():
        raise AdapterError(f"Hermes plugin manifest is missing: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    hooks: set[str] = set()
    in_hooks = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"provides_hooks:", "hooks:"}:
            in_hooks = True
            continue
        if in_hooks and stripped.startswith("-"):
            hooks.add(stripped[1:].strip())
            continue
        if in_hooks and stripped and not stripped.startswith("-"):
            in_hooks = False
    missing = sorted(REQUIRED_HOOKS - hooks)
    if missing:
        raise AdapterError(f"Hermes plugin is missing hooks: {', '.join(missing)}")
    record_path = root / "installation-record.json"
    if not record_path.is_file():
        raise AdapterError("Hermes installation record is missing")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdapterError("Hermes installation record is invalid") from exc
    if not isinstance(record, dict) or record.get("target") != "hermes":
        raise AdapterError("Hermes installation record has an invalid target")
    return {"status": "passed", "plugin_root": str(root), "hooks": sorted(hooks)}


__all__ = ["REQUIRED_HOOKS", "verify_hermes_plugin"]
