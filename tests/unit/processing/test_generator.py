from __future__ import annotations

import pytest

from ledgermind_local.processing.generator import (
    BrokerHypothesisGenerator,
    HypothesisCandidate,
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
        tool_interactions=(),
        normalized_digest="sha256:" + "a" * 64,
        source_event_ids=("event-1", "event-2"),
    )


def test_null_generator_returns_no_knowledge_without_model_call() -> None:
    generator = NullHypothesisGenerator()

    assert generator.generate(_round()) == ()
    assert generator.provider == "none"
    assert generator.model == "none"


def test_hypothesis_candidate_rejects_empty_semantic_fields() -> None:
    with pytest.raises(ValueError, match="title"):
        HypothesisCandidate(title="", target="target", statement="statement")


def test_hypothesis_candidate_validates_source_event_ids_against_round() -> None:
    candidate = HypothesisCandidate(
        title="Title",
        target="Target",
        statement="Statement",
        source_event_ids=("event-1",),
    )

    candidate.validate_source_events(_round())

    invalid = HypothesisCandidate(
        title="Title",
        target="Target",
        statement="Statement",
        source_event_ids=("missing",),
    )
    with pytest.raises(ValueError, match="source_event_ids"):
        invalid.validate_source_events(_round())


def test_broker_generator_delegates_and_exposes_model_metadata() -> None:
    class Broker:
        def __init__(self) -> None:
            self.calls: list[tuple[NormalizedRound, str]] = []

        def generate_hypotheses(
            self, normalized_round: NormalizedRound, profile_id: str
        ) -> tuple[HypothesisCandidate, ...]:
            self.calls.append((normalized_round, profile_id))
            return ()

    broker = Broker()
    generator = BrokerHypothesisGenerator(
        broker,
        profile_id="hypothesis-default",
        provider="openai_compatible",
        model="model",
    )

    assert generator.generate(_round()) == ()
    assert broker.calls == [(_round(), "hypothesis-default")]
    assert generator.prompt_version == 1
    assert generator.schema_version == 1
