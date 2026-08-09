"""Local technical embedding backend boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Vectorizer(Protocol):
    """Protocol implemented by Local embedding backends."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def close(self) -> None: ...
