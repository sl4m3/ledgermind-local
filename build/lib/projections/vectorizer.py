"""Local vectorizer boundary used by derivative vector projections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class Vectorizer(Protocol):
    """Protocol for embedding services used only by local projections."""

    @property
    def fingerprint(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def encode(self, texts: Sequence[str]) -> "npt.NDArray[np.float32]":
        ...

    def close(self) -> None:
        ...
