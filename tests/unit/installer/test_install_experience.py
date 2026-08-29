from __future__ import annotations

import base64
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from ledgermind_local.installer import cli, wizard
from ledgermind_local.installer.config_writer import (
    load_installer_config,
    write_installer_config,
)
from ledgermind_local.installer.models import (
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
)
from ledgermind_local.installer.operations.common import unpack_bundle
from ledgermind_local.installer.operations.integrations import (
    connect_integration,
    disconnect_integration,
    set_integration_enabled,
)
from ledgermind_local.installer.paths import InstallerPaths
from ledgermind_local.installer.targets import registry
from ledgermind_local.installer.targets.base import BaseTargetAdapter, TargetDiscovery


def _config() -> InstallerConfig:
    return InstallerConfig(
        semantic_language="en",
        generation=GenerationConfig(
            endpoint="https://provider.example/v1",
            token="generation-secret",
            model="generation",
            object_resolution_model="resolution",
        ),
        embedding=EmbeddingConfig(
            mode="api",
            api=EmbeddingApiConfig(
                endpoint="https://provider.example/v1",
                token="embedding-secret",
                model="embedding",
                dimensions=3,
            ),
        ),
    )


def test_schema_v1_target_migrates_to_v2_integration() -> None:
    payload = _config().model_dump(mode="json")
    payload["schema_version"] = 1
    payload["target"] = "hermes"
    payload.pop("integrations", None)

    migrated = InstallerConfig.model_validate(payload)

    assert migrated.schema_version == 2
    assert [(item.id, item.enabled) for item in migrated.integrations] == [
        ("hermes", True)
    ]


def test_new_config_does_not_connect_an_agent_implicitly() -> None:
    assert _config().schema_version == 2
    assert _config().integrations == ()


def test_python_312_compatible_zstandard_bundle_unpack(tmp_path: Path) -> None:
    import zstandard

    source = tmp_path / "source"
    source.mkdir()
    (source / "marker.txt").write_text("bundle", encoding="utf-8")
    tar_path = tmp_path / "bundle.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname="bundle")
    archive_path = tmp_path / "bundle.tar.zst"
    archive_path.write_bytes(
        zstandard.ZstdCompressor().compress(tar_path.read_bytes())
    )

    unpacked = unpack_bundle(archive_path, tmp_path / "unpacked")

    assert (unpacked / "marker.txt").read_text(encoding="utf-8") == "bundle"


def test_non_interactive_stdin_tokens_reach_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _config().model_dump(mode="json")
    payload["generation"].pop("token")
    payload["generation"]["token_stdin"] = True
    payload["embedding"]["api"].pop("token")
    payload["embedding"]["api"]["token_stdin"] = True
    source = tmp_path / "install.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(source, 0o600)
    captured: dict[str, Any] = {}

    def fake_install(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(cli, "install", fake_install)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("generation-stdin\nembedding-stdin\n"))

    exit_code = cli.main(
        ["install", "--non-interactive", "--config", str(source), "--json"]
    )

    assert exit_code == 0
    assert captured["generation_stdin"] == "generation-stdin"
    assert captured["embedding_stdin"] == "embedding-stdin"


def test_no_arguments_starts_the_interactive_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_install(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        assert kwargs["config"].semantic_language == "en"
        return {"status": "success"}

    monkeypatch.setattr(wizard, "build_interactive_config", _config)
    monkeypatch.setattr(cli, "install", fake_install)

    assert cli.main([]) == 0
    assert called is True


class _FakeHermesAdapter(BaseTargetAdapter):
    id = "hermes"
    label = "Hermes"

    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self) -> TargetDiscovery:
        return TargetDiscovery("hermes", "Hermes", True, home=self.root)

    def install(self, context: Any) -> dict[str, Any]:
        del context
        plugin = self.root / "plugins" / "ledgermind-hermes"
        plugin.mkdir(parents=True, exist_ok=True)
        encoded = b'{"enabled": true}\n'
        (plugin / "config.json").write_bytes(encoded)
        (plugin / "installation-record.json").write_text(
            json.dumps(
                {
                    "target": "hermes",
                    "files": {
                        "config.json": {
                            "after": {
                                "content": base64.b64encode(encoded).decode("ascii")
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return {"status": "passed"}

    def verify(self, context: Any) -> dict[str, Any]:
        del context
        return {"status": "passed"}

    def uninstall(self, context: Any, *, purge: bool = False) -> dict[str, Any]:
        del context, purge
        return {"status": "passed"}

    def runtime_environment(self, context: Any) -> dict[str, str]:
        del context
        return {
            "LEDGERMIND_HERMES_CONFIG": str(
                self.root / "plugins" / "ledgermind-hermes" / "config.json"
            )
        }


class _FailingHermesAdapter(_FakeHermesAdapter):
    rolled_back = False

    def verify(self, context: Any) -> dict[str, Any]:
        del context
        raise RuntimeError("verification failed")

    def uninstall(self, context: Any, *, purge: bool = False) -> dict[str, Any]:
        del context, purge
        self.rolled_back = True
        return {"status": "passed"}


def test_integration_lifecycle_is_independent_from_platform_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGERMIND_SECRET_BACKEND", "file")
    paths = InstallerPaths(home_override=tmp_path / "install")
    write_installer_config(_config(), paths)
    adapter = _FakeHermesAdapter(tmp_path / "hermes")
    monkeypatch.setitem(registry._ADAPTERS, "hermes", adapter)

    connected = connect_integration(paths=paths, target_id="hermes")
    set_integration_enabled(paths=paths, target_id="hermes", enabled=False)
    disabled = load_installer_config(paths.config_file)
    disconnected = disconnect_integration(paths=paths, target_id="hermes")
    final = load_installer_config(paths.config_file)

    assert connected["connected"] is True
    assert disabled.integrations[0].enabled is False
    assert disconnected["connected"] is False
    assert final.integrations == ()


def test_failed_integration_is_rolled_back_and_not_marked_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGERMIND_SECRET_BACKEND", "file")
    paths = InstallerPaths(home_override=tmp_path / "install")
    write_installer_config(_config(), paths)
    adapter = _FailingHermesAdapter(tmp_path / "hermes")
    monkeypatch.setitem(registry._ADAPTERS, "hermes", adapter)

    with pytest.raises(RuntimeError, match="verification failed"):
        connect_integration(paths=paths, target_id="hermes")

    assert adapter.rolled_back is True
    assert load_installer_config(paths.config_file).integrations == ()
