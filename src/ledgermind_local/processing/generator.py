"""Dependency-injected hypothesis generation boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import NormalizedRound


@dataclass(frozen=True, slots=True)
class HypothesisDraft:
    title: str
    target: str
    statement: str
    rationale: str = ""
    result: str = ""
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("title", "target", "statement"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.rationale, str) or not isinstance(self.result, str):
            raise TypeError("rationale and result must be strings")
        if len(self.title) > 240 or len(self.target) > 240:
            raise ValueError("title and target are too long")
        if len(self.statement) > 20_000 or len(self.rationale) > 40_000 or len(self.result) > 20_000:
            raise ValueError("hypothesis text is too long")
        if len(self.artifacts) > 500 or any(not isinstance(item, str) for item in self.artifacts):
            raise ValueError("artifacts must be a sequence of strings")


class HypothesisGenerator(Protocol):
    provider: str
    model: str
    prompt_version: int
    schema_version: int

    def generate(self, normalized_round: NormalizedRound) -> Sequence[HypothesisDraft]:
        ...


class NullHypothesisGenerator:
    """Explicit no-op generator used when semantic extraction is disabled."""

    provider = "none"
    model = "none"
    prompt_version = 1
    schema_version = 1

    def generate(self, normalized_round: NormalizedRound) -> tuple[HypothesisDraft, ...]:
        del normalized_round
        return ()


class CallableHypothesisGenerator:
    """Small adapter for a model provider owned by the service boundary."""

    def __init__(
        self,
        callback: Callable[[NormalizedRound], Sequence[HypothesisDraft]],
        *,
        provider: str,
        model: str,
        prompt_version: int = 1,
        schema_version: int = 1,
    ) -> None:
        if not provider.strip() or not model.strip():
            raise ValueError("provider and model must not be empty")
        self._callback = callback
        self.provider = provider
        self.model = model
        self.prompt_version = max(int(prompt_version), 1)
        self.schema_version = max(int(schema_version), 1)

    def generate(self, normalized_round: NormalizedRound) -> Sequence[HypothesisDraft]:
        return self._callback(normalized_round)
