from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SandboxLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class SandboxUnavailableError(RuntimeError):
    """Required network isolation is not available on this host."""


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    command: tuple[str, ...]
    level: SandboxLevel
    detail: str


def build_sandbox_plan(
    command: Sequence[str],
    *,
    core_data_dir: Path,
    blocked_data_dirs: Sequence[Path] = (),
    required: bool,
) -> SandboxPlan:
    """Build a Core command with a verified network namespace when available."""

    original = tuple(str(part) for part in command)
    attempts: list[str] = []

    bwrap = shutil.which("bwrap")
    if bwrap:
        prefix = (
            bwrap,
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
        )
        blocked_mounts = tuple(
            item
            for blocked_data_dir in blocked_data_dirs
            for item in ("--tmpfs", str(blocked_data_dir))
        )
        prefix += blocked_mounts + (
            "--bind",
            str(core_data_dir),
            str(core_data_dir),
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--chdir",
            str(core_data_dir),
            "--unsetenv",
            "PWD",
            "--",
        )
        if _probe_command(prefix + ("/bin/true",), core_data_dir):
            return SandboxPlan(
                command=prefix + original,
                level=SandboxLevel.FULL,
                detail="bwrap network namespace",
            )
        attempts.append("bwrap probe failed")

    unshare = shutil.which("unshare")
    if unshare:
        prefix = (unshare, "--net", "--")
        if _probe_command(prefix + ("/bin/true",), core_data_dir):
            return SandboxPlan(
                command=prefix + original,
                level=SandboxLevel.FULL,
                detail="unshare network namespace",
            )
        attempts.append("unshare probe failed")

    detail = "network isolation unavailable"
    if attempts:
        detail = f"{detail}: {', '.join(attempts)}"
    if required:
        raise SandboxUnavailableError(detail)
    return SandboxPlan(command=original, level=SandboxLevel.PARTIAL, detail=detail)


def _probe_command(command: Sequence[str], core_data_dir: Path) -> bool:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=str(core_data_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


__all__ = [
    "SandboxLevel",
    "SandboxPlan",
    "SandboxUnavailableError",
    "build_sandbox_plan",
]
