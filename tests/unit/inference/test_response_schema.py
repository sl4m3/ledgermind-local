from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgermind_local.inference.schemas import HypothesisResponse


def test_hypothesis_response_accepts_zero_and_multiple_candidates() -> None:
    empty = HypothesisResponse.model_validate_json('{"hypotheses": []}')
    assert empty.hypotheses == ()

    response = HypothesisResponse.model_validate(
        {
            "hypotheses": [
                {
                    "title": "One",
                    "target": "ops",
                    "statement": "One statement",
                    "rationale": "reason",
                    "result": "result",
                    "artifacts": ["artifact"],
                    "source_event_ids": ["event-1"],
                },
                {
                    "title": "Two",
                    "target": "ops",
                    "statement": "Two statement",
                    "rationale": "",
                    "result": "",
                    "artifacts": [],
                    "source_event_ids": ["event-2"],
                },
            ]
        }
    )
    assert len(response.hypotheses) == 2
    assert response.to_candidates()[0].source_event_ids == ("event-1",)


def test_hypothesis_response_rejects_phase_and_invalid_json() -> None:
    payload = {
        "hypotheses": [
            {
                "title": "One",
                "target": "ops",
                "statement": "Statement",
                "rationale": "",
                "result": "",
                "artifacts": [],
                "source_event_ids": ["event-1"],
                "phase": "pattern",
            }
        ]
    }
    with pytest.raises(ValueError):
        HypothesisResponse.model_validate(payload)
    with pytest.raises(ValueError):
        HypothesisResponse.model_validate_json("not-json")


def test_versioned_prompts_and_schema_contract_are_packaged() -> None:
    root = Path(__file__).parents[3] / "src" / "ledgermind_local" / "inference"
    hypothesis_prompt = (root / "prompts" / "hypothesis_v1.txt").read_text(
        encoding="utf-8"
    )
    schema = json.loads(
        (root / "schemas" / "hypothesis-response-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert "phase" in hypothesis_prompt
    assert schema["additionalProperties"] is False
    assert "phase" not in schema["properties"]["hypotheses"]["items"]["properties"]
