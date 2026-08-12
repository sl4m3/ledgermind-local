"""Offline-friendly, atomic artifact downloads."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .errors import DownloadError
from .installer_types import DownloadedArtifact
from .verify import sha256_file


def _source_path(url: str) -> Path | None:
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse

        return Path(unquote(urlparse(url).path)).expanduser()
    candidate = Path(url).expanduser()
    return candidate if candidate.exists() else None


def download_artifact(
    url: str,
    destination: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> DownloadedArtifact:
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
        ) as handle:
            temporary = Path(handle.name)
            source = _source_path(url)
            if source is not None:
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_size is not None and temporary.stat().st_size != expected_size:
            raise DownloadError("downloaded artifact size does not match manifest")
        digest = sha256_file(temporary)
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise DownloadError("downloaded artifact SHA-256 does not match manifest")
        temporary.replace(target)
        return DownloadedArtifact(
            path=target, size=target.stat().st_size, sha256=digest
        )
    except DownloadError:
        raise
    except (OSError, ValueError, TimeoutError) as exc:
        raise DownloadError(f"failed to download artifact: {url}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["download_artifact"]
