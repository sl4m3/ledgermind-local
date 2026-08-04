from __future__ import annotations

from pathlib import Path

import pytest

from ledgermind_local.core_gateway.sandbox import (
    SandboxLevel,
    SandboxUnavailableError,
    build_sandbox_plan,
)


def test_missing_network_sandbox_is_partial_when_not_required(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox.shutil.which",
        lambda _: None,
    )

    plan = build_sandbox_plan(
        ("/opt/ledgermind-core/bin/ledgermind-core", "--database", "knowledge.db"),
        core_data_dir=tmp_path,
        required=False,
    )

    assert plan.level is SandboxLevel.PARTIAL
    assert plan.command == (
        "/opt/ledgermind-core/bin/ledgermind-core",
        "--database",
        "knowledge.db",
    )


def test_missing_required_network_sandbox_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox.shutil.which",
        lambda _: None,
    )

    with pytest.raises(SandboxUnavailableError):
        build_sandbox_plan(
            ("/opt/ledgermind-core/bin/ledgermind-core",),
            core_data_dir=tmp_path,
            required=True,
        )


def test_bwrap_plan_is_full_only_after_probe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox._probe_command",
        lambda command, core_data_dir: command[0] == "/usr/bin/bwrap",
    )

    plan = build_sandbox_plan(
        ("/opt/ledgermind-core/bin/ledgermind-core", "--database", "knowledge.db"),
        core_data_dir=tmp_path,
        required=True,
    )

    assert plan.level is SandboxLevel.FULL
    assert plan.command[:2] == ("/usr/bin/bwrap", "--unshare-net")
    assert plan.command[-4:] == (
        "--",
        "/opt/ledgermind-core/bin/ledgermind-core",
        "--database",
        "knowledge.db",
    )


def test_bwrap_plan_masks_blocked_local_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "ledgermind_local.core_gateway.sandbox._probe_command",
        lambda command, core_data_dir: True,
    )
    local_data_dir = tmp_path / "local"

    plan = build_sandbox_plan(
        ("/opt/ledgermind-core/bin/ledgermind-core",),
        core_data_dir=tmp_path / "core",
        blocked_data_dirs=(local_data_dir,),
        required=True,
    )

    assert plan.level is SandboxLevel.FULL
    assert plan.command[plan.command.index("--tmpfs") + 1] == str(local_data_dir)
