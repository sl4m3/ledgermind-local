"""Wheel content checks for the clean local package."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

_FORBIDDEN_TOP_LEVEL = {
    "application",
    "contracts",
    "domain",
    "ports",
    "api",
    "persistence",
}
_FORBIDDEN_COMPONENTS = {
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_FORBIDDEN_SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".token"}
_FORBIDDEN_DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_REQUIRED_RUNTIME_FILES = {
    "ledgermind_local.persistence/rounds_migrations/0001_initial.sql",
    "ledgermind_local.persistence/rounds_migrations/0002_projection_state_compat.sql",
    "ledgermind_local.persistence/rounds_migrations/0003_core_projection_inbox.sql",
    "ledgermind_local.persistence/rounds_migrations/0004_core_projection_fts.sql",
    "ledgermind_local.inference/prompts/hypothesis_v1.txt",
    "ledgermind_local.inference/prompts/merge_v1.txt",
    "ledgermind_local.inference/schemas/hypothesis-response-v1.schema.json",
    "ledgermind_local.inference/schemas/merge-response-v1.schema.json",
}


def _build_wheel_names(source: Path, tmp_path: Path) -> list[str]:
    checkout = tmp_path / "checkout"
    shutil.copytree(
        source,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheel_dir),
            ".",
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    with ZipFile(wheel) as archive:
        return archive.namelist()


def test_clean_local_wheel_contains_only_namespaced_package(tmp_path: Path) -> None:
    names = _build_wheel_names(Path(__file__).resolve().parents[2], tmp_path)
    top_level = {name.split("/", 1)[0] for name in names if "/" in name}
    assert "ledgermind_local" in top_level
    assert not top_level & _FORBIDDEN_TOP_LEVEL


def test_local_wheel_contains_runtime_data_and_no_release_artifacts(
    tmp_path: Path,
) -> None:
    names = _build_wheel_names(Path(__file__).resolve().parents[2], tmp_path)
    normalized = {name.replace("/", ".", 1) if name.startswith("ledgermind_local/") else name for name in names}
    paths = {Path(name) for name in names}

    assert _REQUIRED_RUNTIME_FILES <= normalized
    assert not any(
        component in _FORBIDDEN_COMPONENTS
        for path in paths
        for component in path.parts
    )
    assert not any(path.suffix.lower() == ".pyc" for path in paths)
    assert not any(path.parts[0] == "tests" for path in paths)
    assert not any(
        path.suffix.lower() in _FORBIDDEN_DATABASE_SUFFIXES
        for path in paths
    )
    forbidden_secret_names = {
        ".env",
        ".env.example",
        "credentials",
        "credentials.json",
        "private_key",
        "private_key.json",
        "secret",
        "secret.json",
    }
    assert not any(
        path.name.lower() in forbidden_secret_names
        or path.suffix.lower() in _FORBIDDEN_SECRET_SUFFIXES
        for path in paths
    )
    assert not any(re.search(r"(^|/)build-copy(/|$)", name) for name in names)
