from __future__ import annotations

from pathlib import Path

from ledgermind_local.installer.models import (
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
)
from ledgermind_local.installer.paths import InstallerPaths
from ledgermind_local.installer.targets.base import AdapterContext
from ledgermind_local.installer.targets.hermes.adapter import HermesTargetAdapter


def _config() -> InstallerConfig:
    return InstallerConfig(
        semantic_language="en",
        generation=GenerationConfig(
            endpoint="https://provider.example/v1", token="token", model="model"
        ),
        embedding=EmbeddingConfig(
            mode="api",
            api=EmbeddingApiConfig(
                endpoint="https://provider.example/v1",
                token="token",
                model="embedding",
                dimensions=3,
            ),
        ),
    )


def test_hermes_adapter_restores_only_owned_files(tmp_path: Path, monkeypatch) -> None:
    hermes = tmp_path / "hermes"
    (hermes / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    bundle = tmp_path / "bundle" / "integrations" / "hermes" / "plugin"
    bundle.mkdir(parents=True)
    (bundle / "plugin.yaml").write_text(
        "name: ledgermind-hermes\n"
        "provides_hooks:\n"
        "  - pre_llm_call\n"
        "  - pre_tool_call\n"
        "  - post_tool_call\n"
        "  - post_llm_call\n"
        "  - on_session_end\n",
        encoding="utf-8",
    )
    (bundle / "__init__.py").write_text("register = object\n", encoding="utf-8")
    context = AdapterContext(
        config=_config(),
        paths=InstallerPaths(home_override=tmp_path / "install"),
        bundle_root=tmp_path / "bundle",
        discovery=HermesTargetAdapter().discover(),
    )
    adapter = HermesTargetAdapter()
    installed = adapter.install(context)
    assert installed["files"] == 3
    plugin = hermes / "plugins" / "ledgermind-hermes"
    extra = plugin / "user-notes.txt"
    extra.write_text("keep", encoding="utf-8")
    (plugin / "plugin.yaml").write_text("user changed", encoding="utf-8")
    result = adapter.uninstall(context)
    assert result["skipped_user_changes"] == 1
    assert extra.read_text(encoding="utf-8") == "keep"
    assert (plugin / "plugin.yaml").read_text(encoding="utf-8") == "user changed"
    assert not (plugin / "__init__.py").exists()
    assert not (plugin / "config.json").exists()
