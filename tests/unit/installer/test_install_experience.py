from __future__ import annotations

import base64
import importlib
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
from ledgermind_local.installer.errors import ProviderProbeError
from ledgermind_local.installer.models import (
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
    IntegrationConfig,
)
from ledgermind_local.installer.operations.common import unpack_bundle
from ledgermind_local.installer.operations.integrations import (
    connect_integration,
    disconnect_integration,
    set_integration_enabled,
)
from ledgermind_local.installer.operations.uninstall import uninstall
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


def test_legacy_target_migrates_to_current_integration() -> None:
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
    assert _config().memory_mode == "per_agent"


def test_memory_mode_resolves_shared_or_per_agent_spaces() -> None:
    config = _config()
    assert config.memory_space_id_for("hermes") == "hermes-default"
    assert config.memory_space_id_for("codex") == "codex-default"

    shared = config.model_copy(update={"memory_mode": "shared"})
    assert shared.memory_space_id_for("hermes") == "shared-default"
    assert shared.memory_space_id_for("codex") == "shared-default"


def test_operation_urls_are_normalized_to_openai_api_base() -> None:
    generation = GenerationConfig(
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        token="secret",
        model="model",
    )
    embedding = EmbeddingApiConfig(
        endpoint="https://openrouter.ai/api/v1/embeddings",
        token="secret",
        model="embedding",
        dimensions=3,
    )

    assert generation.endpoint == "https://openrouter.ai/api/v1"
    assert embedding.endpoint == "https://openrouter.ai/api/v1"


def test_terminal_wizard_uses_reference_openrouter_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DetectedAdapter:
        label = "Codex CLI"

        @staticmethod
        def discover() -> TargetDiscovery:
            return TargetDiscovery("codex", "Codex CLI", True, home=Path("/tmp/codex"))

    monkeypatch.setattr(registry, "target_ids", lambda: ("codex",))
    monkeypatch.setattr(
        registry, "get_target_adapter", lambda _target: _DetectedAdapter()
    )
    answers = iter(
        (
            "",  # Russian
            "",  # OpenRouter
            "",  # reference generation model
            "",  # same Object Resolution model
            "2",  # primary + fallback
            "",  # no provider is selected implicitly
            "baidu/fp8",  # explicit primary provider
            "deepinfra/fp8",  # explicit fallback provider
            "",  # API embeddings
            "",  # same API base
            "",  # reuse token
            "",  # reference embedding model
            "",  # 2048 dimensions
            "2",  # shared memory
            "",  # recommended runtime settings
            "1",  # Codex
            "",  # install
        )
    )
    output = io.StringIO()

    config = wizard.build_interactive_config(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: "secret",
        output=output,
    )

    assert config.semantic_language == "ru"
    assert config.generation.endpoint == "https://openrouter.ai/api/v1"
    assert config.generation.route == "baidu/fp8"
    assert config.generation.fallback_routes == ("deepinfra/fp8",)
    assert config.generation.model == wizard.REFERENCE_GENERATION_MODEL
    assert config.embedding.api is not None
    assert config.embedding.api.endpoint == "https://openrouter.ai/api/v1"
    assert config.embedding.api.model == "nvidia/nemotron-3-embed-1b:free"
    assert config.embedding.api.dimensions == 2048
    assert config.memory_mode == "shared"
    assert [item.id for item in config.integrations] == ["codex"]
    assert "LEDGERMIND SETUP" in output.getvalue()
    assert "REVIEW" in output.getvalue()
    assert "A value is required." in output.getvalue()
    assert "baidu/fp8 → deepinfra/fp8 (restricted)" in output.getvalue()


def test_install_fails_provider_preflight_before_release_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module = importlib.import_module(
        "ledgermind_local.installer.operations.install"
    )
    fetched = False

    def fail_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ProviderProbeError("configured generation model is unavailable")

    def unexpected_fetch(*_args: Any, **_kwargs: Any) -> tuple[Path, Path, Path]:
        nonlocal fetched
        fetched = True
        raise AssertionError("release download must not start")

    monkeypatch.setattr(install_module, "probe_generation", fail_probe)
    monkeypatch.setattr(install_module, "fetch_release", unexpected_fetch)

    with pytest.raises(ProviderProbeError, match="unavailable"):
        install_module.install(
            config=_config(), paths=InstallerPaths(home_override=tmp_path)
        )

    assert fetched is False


def test_python_312_compatible_zstandard_bundle_unpack(tmp_path: Path) -> None:
    import zstandard

    source = tmp_path / "source"
    source.mkdir()
    (source / "marker.txt").write_text("bundle", encoding="utf-8")
    tar_path = tmp_path / "bundle.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname="bundle")
    archive_path = tmp_path / "bundle.tar.zst"
    archive_path.write_bytes(zstandard.ZstdCompressor().compress(tar_path.read_bytes()))

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
    monkeypatch.setattr(
        cli.sys, "stdin", io.StringIO("generation-stdin\nembedding-stdin\n")
    )

    exit_code = cli.main(
        [
            "install",
            "--home",
            str(tmp_path / "install-home"),
            "--non-interactive",
            "--config",
            str(source),
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["generation_stdin"] == "generation-stdin"
    assert captured["embedding_stdin"] == "embedding-stdin"


def test_no_arguments_starts_the_interactive_install(
    tmp_path: Path,
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
    monkeypatch.setattr(
        cli,
        "_paths",
        lambda _args: InstallerPaths(home_override=tmp_path / "install-home"),
    )

    assert cli.main([]) == 0
    assert called is True


def test_repeated_install_adds_agent_without_reinstalling_or_reconfiguring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGERMIND_SECRET_BACKEND", "file")
    paths = InstallerPaths(home_override=tmp_path / "install-home")
    write_installer_config(_config(), paths)
    connected: list[str] = []

    def fake_connect(**kwargs: Any) -> dict[str, Any]:
        connected.append(str(kwargs["target_id"]))
        return {"status": "passed", "connected": True}

    monkeypatch.setattr(cli, "connect_integration", fake_connect)
    monkeypatch.setattr(
        cli,
        "install",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing install must not download or reinstall")
        ),
    )

    exit_code = cli.main(
        [
            "install",
            "--home",
            str(tmp_path / "install-home"),
            "--existing-mode",
            "add-agent",
            "--agent",
            "hermes",
            "--json",
        ]
    )

    assert exit_code == 0
    assert connected == ["hermes"]


def test_update_uses_installed_provider_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEDGERMIND_SECRET_BACKEND", "file")
    paths = InstallerPaths(home_override=tmp_path / "install-home")
    expected = _config()
    write_installer_config(expected, paths)
    installed = load_installer_config(paths.config_file)
    captured: dict[str, Any] = {}

    def fake_update(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(cli, "update", fake_update)

    exit_code = cli.main(
        [
            "update",
            "--home",
            str(tmp_path / "install-home"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--bundle",
            str(tmp_path / "bundle.tar.zst"),
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["config"] == installed
    assert captured["generation_stdin"] is None
    assert captured["embedding_stdin"] is None


def test_provider_reconfiguration_preserves_memory_and_agents() -> None:
    existing = _config().model_copy(
        update={
            "memory_mode": "shared",
            "memory_data_path": "/private/memory",
            "integrations": (IntegrationConfig(id="codex", enabled=True),),
        }
    )
    answers = iter(
        (
            "3",  # custom provider
            "https://new-provider.example/v1",
            "new-generation",
            "",  # same OR model
            "",  # API embeddings
            "",  # same endpoint
            "",  # reuse generation token
            "new-embedding",
            "",  # 1536 dimensions
            "",  # apply
        )
    )

    updated = wizard.build_interactive_config(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: "new-secret",
        output=io.StringIO(),
        existing_config=existing,
    )

    assert updated.generation.model == "new-generation"
    assert updated.embedding.api is not None
    assert updated.embedding.api.model == "new-embedding"
    assert updated.semantic_language == existing.semantic_language
    assert updated.memory_mode == "shared"
    assert updated.memory_data_path == "/private/memory"
    assert updated.integrations == existing.integrations
    assert updated.runtime == existing.runtime


def test_default_uninstall_preserves_and_backs_up_memory(tmp_path: Path) -> None:
    paths = InstallerPaths(home_override=tmp_path / "install-home")
    paths.ensure()
    marker = paths.memory_data_dir / "knowledge.db"
    marker.write_bytes(b"opaque-memory")

    result = uninstall(paths=paths)

    assert result["preserved_data"] is True
    assert result["preserved_config"] is True
    assert result["backup_status"] == "created"
    assert marker.read_bytes() == b"opaque-memory"
    archive = Path(result["memory_backup"])
    assert archive.is_file()
    assert archive.stat().st_mode & 0o777 == 0o600
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.extractfile("memory-data/knowledge.db")
        assert member is not None
        assert member.read() == b"opaque-memory"


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
    config_writer = importlib.import_module("ledgermind_local.installer.config_writer")

    def unexpected_profile_rewrite(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError(
            "integration lifecycle must not rewrite inference profiles"
        )

    monkeypatch.setattr(
        config_writer, "write_local_profiles", unexpected_profile_rewrite
    )
    adapter = _FakeHermesAdapter(tmp_path / "hermes")
    monkeypatch.setitem(registry._ADAPTERS, "hermes", adapter)

    connected = connect_integration(paths=paths, target_id="hermes")
    set_integration_enabled(paths=paths, target_id="hermes", enabled=False)
    disabled = load_installer_config(paths.config_file)
    disconnected = disconnect_integration(paths=paths, target_id="hermes")
    final = load_installer_config(paths.config_file)

    assert connected["connected"] is True
    assert connected["summary"] == {
        "label": "Hermes",
        "agent_location": str(tmp_path / "hermes"),
        "connected": True,
        "enabled": True,
        "verification": "passed",
        "activation_required": None,
        "memory_mode": "per_agent",
        "memory_space_id": "hermes-default",
        "inference_profiles_preserved": True,
    }
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
