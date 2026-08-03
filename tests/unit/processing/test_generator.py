from __future__ import annotations

import pytest

from ledgermind_local.processing.generator import (
    HypothesisDraft,
    NullHypothesisGenerator,
)
from ledgermind_local.processing.models import NormalizedRound


def _round() -> NormalizedRound:
    return NormalizedRound(
        memory_space_id="space",
        source_system="hermes",
        source_instance_id="instance",
        source_profile_id="profile",
        source_session_id="session",
        source_round_id="round",
        started_at="2026-08-02T20:00:00Z",
        completed_at="2026-08-02T20:01:00Z",
        user_text="request",
        assistant_text="answer",
        transcript="user: request\nassistant: answer",
        tool_calls=(),
        normalized_digest="sha256:" + "a" * 64,
    )


def test_null_generator_returns_no_knowledge_without_model_call() -> None:
    generator = NullHypothesisGenerator()

    assert generator.generate(_round()) == ()
    assert generator.provider == "none"
    assert generator.model == "none"


def test_hypothesis_draft_rejects_empty_semantic_fields() -> None:
    with pytest.raises(ValueError, match="title"):
        HypothesisDraft(title="", target="target", statement="statement")
