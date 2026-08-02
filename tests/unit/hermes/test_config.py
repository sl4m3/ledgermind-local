"""Tests for plugin config loading."""

from __future__ import annotations

from pathlib import Path

from plugins.hermes.config import load_config


def test_load_config_reads_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        (
            "{\n"
            '  "config_version": 1,\n'
            '  "source_instance_id": "src_01J",\n'
            '  "service_url": "http://127.0.0.1:8000",\n'
            '  "token_file": "~/.ledgermind/local/server.token",\n'
            '  "profile_name": "default",\n'
            '  "memory_space_id": "hermes:src_01J:default",\n'
            '  "state_db_path": "~/.hermes/state.db",\n'
            '  "extraction_prompt_version": 1,\n'
            '  "extraction_schema_version": 1,\n'
            '  "pre_llm_timeout_seconds": 1.0,\n'
            '  "delivery_timeout_seconds": 5.0,\n'
            '  "max_context_items": 5\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.source_instance_id == "src_01J"
    assert config.memory_space_id == "hermes:src_01J:default"
