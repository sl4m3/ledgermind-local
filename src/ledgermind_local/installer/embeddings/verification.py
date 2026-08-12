"""Post-download and runtime embedding verification."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..verify import sha256_file


def verify_model_files(model_dir: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    root = Path(model_dir)
    checked = 0
    expected_files = dict(entry.get("sha256", {}))
    if not expected_files and isinstance(entry.get("files"), list):
        expected_files = {
            str(record.get("name")): record.get("sha256")
            for record in entry["files"]
            if isinstance(record, dict) and record.get("name") and record.get("sha256")
        }
    for name, expected in expected_files.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path).lower() != str(expected).lower():
            raise ValueError(f"model checksum mismatch: {name}")
        checked += 1
    return {
        "files_checked": checked,
        "license": entry.get("license"),
        "status": "passed",
    }


def run_embedding_smoke(
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    *,
    dimensions: int,
) -> dict[str, Any]:
    one = [
        tuple(float(value) for value in vector)
        for vector in embed(("LedgerMind smoke",))
    ]
    batch = [
        tuple(float(value) for value in vector)
        for vector in embed(("LedgerMind smoke", "LedgerMind smoke 2"))
    ]
    if len(one) != 1 or len(batch) != 2:
        raise ValueError("embedding smoke returned an invalid batch")
    if any(
        len(vector) != dimensions or not all(math.isfinite(value) for value in vector)
        for vector in (*one, *batch)
    ):
        raise ValueError(
            "embedding smoke returned invalid dimensions or non-finite floats"
        )
    repeat = tuple(float(value) for value in embed(("LedgerMind smoke",))[0])
    deterministic = repeat == one[0]
    if not deterministic:
        raise ValueError("embedding runtime is not deterministic for identical input")
    return {
        "status": "passed",
        "dimensions": dimensions,
        "deterministic": deterministic,
    }


__all__ = ["run_embedding_smoke", "verify_model_files"]
