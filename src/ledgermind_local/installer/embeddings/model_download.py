"""Download approved Hugging Face model files with per-file hashes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..download import download_artifact
from ..errors import ConfigurationError
from ..paths import InstallerPaths


def _file_records(entry: dict[str, Any]) -> list[dict[str, Any]]:
    records = entry.get("files", [])
    if isinstance(records, dict):
        return [
            {"name": name, "url": url, "sha256": entry.get("sha256", {}).get(name)}
            for name, url in records.items()
        ]
    if not isinstance(records, list):
        raise ConfigurationError("embedding catalog files must be a list")
    result: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, str):
            result.append({"name": Path(record).name, "url": record})
        elif isinstance(record, dict):
            result.append(dict(record))
        else:
            raise ConfigurationError(
                "embedding catalog contains an invalid file record"
            )
    hashes = entry.get("sha256", {})
    for record in result:
        record.setdefault("sha256", hashes.get(record.get("name")))
    return result


def download_model(entry: dict[str, Any], paths: InstallerPaths) -> Path:
    model_id = str(entry.get("id", "")).strip()
    if not model_id:
        raise ConfigurationError("embedding catalog entry has no id")
    target_dir = paths.models_dir / model_id
    target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for record in _file_records(entry):
        name = str(record.get("name", "")).strip()
        url = str(record.get("url", "")).strip()
        if not name or not url or Path(name).name != name:
            raise ConfigurationError("embedding model file name or URL is invalid")
        destination = target_dir / name
        download_artifact(
            url,
            destination,
            expected_size=int(record["size"])
            if record.get("size") is not None
            else None,
            expected_sha256=str(record["sha256"]) if record.get("sha256") else None,
        )
    return target_dir


__all__ = ["download_model"]
