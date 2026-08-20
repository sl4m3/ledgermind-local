from __future__ import annotations

import json
import stat
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ledgermind_local.installer.config_writer import (
    load_installer_config,
    write_installer_config,
    write_local_config,
    write_local_profiles,
)
from ledgermind_local.installer.embeddings.service import EmbeddingService
from ledgermind_local.installer.errors import SignatureVerificationError
from ledgermind_local.installer.models import (
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
)
from ledgermind_local.installer.paths import InstallerPaths
from ledgermind_local.installer.verify import verify_ed25519
from ledgermind_local.runtime.supervisor import RuntimeSupervisor


def _config() -> InstallerConfig:
    return InstallerConfig(
        semantic_language="en",
        generation=GenerationConfig(
            endpoint="https://provider.example/v1",
            token="generation-secret",
            model="generation-model",
            object_resolution_model="object-resolution-model",
        ),
        embedding=EmbeddingConfig(
            mode="api",
            api=EmbeddingApiConfig(
                endpoint="https://provider.example/v1",
                token="embedding-secret",
                model="embedding-model",
                dimensions=3,
            ),
        ),
    )


def test_config_writer_uses_0600_file_fallback_and_strips_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGERMIND_SECRET_BACKEND", "file")
    paths = InstallerPaths(home_override=tmp_path)
    write_installer_config(_config(), paths)

    loaded = load_installer_config(paths.config_file)
    assert loaded.generation.secret_ref == "generation/token"
    assert loaded.generation.token is None
    assert "generation-secret" not in paths.config_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(paths.config_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.secrets_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.secrets_file.parent.stat().st_mode) == 0o700


def test_local_service_config_is_written_under_xdg_data(tmp_path: Path) -> None:
    paths = InstallerPaths(home_override=tmp_path)
    write_local_config(_config(), paths)

    service_config = paths.data_dir / "local" / "config.json"
    assert service_config.is_file()
    assert stat.S_IMODE(service_config.stat().st_mode) == 0o600


def test_installer_materializes_profiles_for_local_resolver(tmp_path: Path) -> None:
    from ledgermind_local.inference.profile_store import InferenceProfileStore
    from ledgermind_local.persistence import open_sqlite_connection

    paths = InstallerPaths(home_override=tmp_path)
    result = write_local_profiles(_config(), paths)
    connection = open_sqlite_connection(result["database"])
    try:
        slots = InferenceProfileStore(connection).list_slots("hermes-default")
    finally:
        connection.close()

    assert slots == {
        "background": "generation-background",
        "embedding": "embedding-default",
        "operational": "generation-operational",
        "object_resolution": "generation-object-resolution",
    }


def test_ed25519_rejects_tampered_manifest() -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload = b"signed manifest"
    verify_ed25519(payload, private.sign(payload), public)
    with pytest.raises(SignatureVerificationError):
        verify_ed25519(b"tampered", private.sign(payload), public)


def test_runtime_supports_multiple_leases_and_ttl_cleanup(
    tmp_path: Path,
) -> None:
    supervisor = RuntimeSupervisor(
        InstallerPaths(home_override=tmp_path),
        lease_ttl_seconds=0.1,
        idle_shutdown_seconds=0,
    )
    first = supervisor.acquire(client="hermes", session_id="one")
    second = supervisor.acquire(client="hermes", session_id="two")
    assert first["started"] is True
    assert second["started"] is False
    assert len(supervisor.status()["active_leases"]) == 2
    time.sleep(0.15)
    assert supervisor.status()["active_leases"] == []
    assert supervisor.status()["running"] is False


def test_local_embedding_service_is_openai_compatible() -> None:
    service = EmbeddingService(
        backend=lambda texts: [[float(len(text)), 1.0, 0.0] for text in texts],
        model="local-model",
        dimensions=3,
        device="cpu",
    )
    endpoint = service.start()
    try:
        request = Request(
            endpoint + "/embeddings",
            data=json.dumps({"model": "local-model", "input": ["hello"]}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode())
        assert payload["data"][0]["embedding"] == [5.0, 1.0, 0.0]
    finally:
        service.stop()
