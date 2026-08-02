"""Tests for bootstrap projection selection."""

from __future__ import annotations

from ledgermind_local.bootstrap import build_projection_names
from ledgermind_local.config import LocalConfig


def test_build_projection_names_without_markdown_audit() -> None:
    config = LocalConfig(config_version=1)
    projection_names = build_projection_names(config)
    assert projection_names == (
        "projections.search",
        "projections.knowledge",
        "projections.markdown",
    )


def test_build_projection_names_with_markdown_audit() -> None:
    config = LocalConfig(config_version=1, markdown_audit_enabled=True)
    projection_names = build_projection_names(config)
    assert projection_names == (
        "projections.search",
        "projections.knowledge",
        "projections.markdown",
        "projections.markdown_audit",
    )
