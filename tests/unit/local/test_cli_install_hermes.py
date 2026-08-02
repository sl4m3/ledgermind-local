"""Tests for `ledgermind install hermes`."""

from __future__ import annotations

import json

import yaml

import cli
from cli import main


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def health(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return {"status": "ok"}

    def search_context(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return {
            "items": [],
            "count": 0,
        }


def test_install_hermes_creates_profile_plugin_layout(tmp_path, monkeypatch: object) -> None:
    home = tmp_path / "service"
    assert main(["--home", str(home), "init"]) == 0

    hermes_home = tmp_path / "hermes"
    monkeypatch.setattr(cli, "_resolve_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(cli, "LedgerMindClient", _FakeClient)

    assert main(["--home", str(home), "install", "hermes"]) == 0

    profile_home = hermes_home / "profiles" / "default"
    plugin_dir = profile_home / "plugins" / "ledgermind"
    assert (plugin_dir / "plugin.yaml").exists()
    assert (plugin_dir / "config.json").exists()

    payload = json.loads((plugin_dir / "config.json").read_text(encoding="utf-8"))
    assert payload["memory_space_id"].startswith("hermes:src_")
    assert payload["profile_name"] == "default"

    profile_config = yaml.safe_load((profile_home / "config.yaml").read_text(encoding="utf-8"))
    plugins = profile_config["plugins"]
    assert "ledgermind" in plugins["enabled"]
