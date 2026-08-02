"""Local vectorizer boundary used by derivative vector projections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Vectorizer(Protocol):
    """Protocol for embedding services used only by local projections."""

    @property
    def fingerprint(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    def close(self) -> None:
        ...
