"""Versioned structured response models for inference tasks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..processing.generator import HypothesisCandidate


class HypothesisResponseItem(BaseModel):
    """One model-produced candidate before Local persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=240)
    target: str = Field(min_length=1, max_length=240)
    statement: str = Field(min_length=1, max_length=20_000)
    rationale: str = Field(default="", max_length=40_000)
    result: str = Field(default="", max_length=20_000)
    artifacts: tuple[str, ...] = Field(default=(), max_length=500)
    source_event_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)

    def to_candidate(self) -> HypothesisCandidate:
        return HypothesisCandidate(
            title=self.title,
            target=self.target,
            statement=self.statement,
            rationale=self.rationale,
            result=self.result,
            artifacts=self.artifacts,
            source_event_ids=self.source_event_ids,
        )


class HypothesisResponse(BaseModel):
    """Strict hypothesis response; zero hypotheses is valid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypotheses: tuple[HypothesisResponseItem, ...] = Field(default=(), max_length=500)

    def to_candidates(self) -> tuple[HypothesisCandidate, ...]:
        return tuple(item.to_candidate() for item in self.hypotheses)


class MergeProposal(BaseModel):
    """Strict provider proposal for one Core merge task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=240)
    target: str = Field(min_length=1, max_length=240)
    statement: str = Field(min_length=1, max_length=20_000)
    rationale: str = Field(default="", max_length=40_000)
    preserved_references: tuple[str, ...] = Field(min_length=2, max_length=500)
    preserved_constraints: tuple[str, ...] = Field(default=(), max_length=500)


MergeResponse = MergeProposal


__all__ = [
    "HypothesisResponse",
    "HypothesisResponseItem",
    "MergeProposal",
    "MergeResponse",
]
