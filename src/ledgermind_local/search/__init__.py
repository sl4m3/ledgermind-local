"""Search helpers and adapters for local knowledge retrieval."""

from .core_backed import CoreBackedSearch
from .fts import CoreProjectionSearchAdapter

__all__ = [
    "CoreBackedSearch",
    "CoreProjectionSearchAdapter",
]
