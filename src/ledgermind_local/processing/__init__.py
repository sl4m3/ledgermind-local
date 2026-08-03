"""Processing package for RawRound normalization and hypothesis generation."""

from .bridge import CoreHypothesisBridge
from .generator import (
    CallableHypothesisGenerator,
    HypothesisDraft,
    HypothesisGenerator,
    NullHypothesisGenerator,
)
from .models import NormalizedRound, NormalizedToolCall
from .normalizer import normalize_raw_round, redact_text, redact_value
from .worker import ProcessingResult, RoundProcessingWorker

__all__ = [
    "CallableHypothesisGenerator",
    "CoreHypothesisBridge",
    "HypothesisDraft",
    "HypothesisGenerator",
    "NormalizedRound",
    "NormalizedToolCall",
    "NullHypothesisGenerator",
    "ProcessingResult",
    "RoundProcessingWorker",
    "normalize_raw_round",
    "redact_text",
    "redact_value",
]
