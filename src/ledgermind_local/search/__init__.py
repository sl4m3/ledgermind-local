"""Search ports, projection adapters, and Core-backed retrieval."""

from .core_backed import (
    DEGRADED_SEARCH_STATUS,
    CandidateScore,
    CoreBackedSearch,
    LocalCandidateSearch,
    SearchStatusCallback,
)
from .fts import CoreProjectionSearchAdapter
from .vector import (
    CoreProjectionVectorSearchAdapter,
    VectorProjectionSearchAdapter,
)

__all__ = [
    "DEGRADED_SEARCH_STATUS",
    "CandidateScore",
    "CoreBackedSearch",
    "CoreProjectionSearchAdapter",
    "CoreProjectionVectorSearchAdapter",
    "LocalCandidateSearch",
    "SearchStatusCallback",
    "VectorProjectionSearchAdapter",
]
