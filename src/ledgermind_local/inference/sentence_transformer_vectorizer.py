"""SentenceTransformers adapter for the signed Nemotron embedding package."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from .vectorizer import EmbeddingRole, Vectorizer

_ModelBuilder: TypeAlias = Callable[[str, str], Any]


def _default_builder(model_path: str, device: str) -> Any:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required by the signed Nemotron runtime"
        ) from exc
    return SentenceTransformer(
        model_path,
        device=device,
        local_files_only=True,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )


class SentenceTransformerVectorizer(Vectorizer):
    """Role-aware local vectorizer using encode_query/encode_document."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str,
        expected_dimension: int = 2048,
        model_builder: _ModelBuilder | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise ValueError(f"local embedding model does not exist: {path}")
        if device not in {"cpu", "cuda", "rocm"}:
            raise ValueError(f"unsupported local embedding device: {device}")
        # PyTorch uses the CUDA device API for both NVIDIA CUDA and AMD ROCm.
        runtime_device = "cuda" if device in {"cuda", "rocm"} else "cpu"
        self._path = path
        self._model = (model_builder or _default_builder)(str(path), runtime_device)
        self._model.max_seq_length = 32768
        dimension = self._model.get_sentence_embedding_dimension()
        self._dimension = int(dimension)
        if self._dimension != expected_dimension:
            raise ValueError(
                f"local embedding dimension mismatch: expected {expected_dimension}, "
                f"received {self._dimension}"
            )
        self._fingerprint = self._tree_fingerprint(path)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        role: EmbeddingRole | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        values = list(texts)
        if role == "query":
            encoded = self._model.encode_query(
                values,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            encoded = self._model.encode_document(
                values,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        vectors = [[float(value) for value in row] for row in rows]
        if len(vectors) != len(values) or any(
            len(vector) != self._dimension for vector in vectors
        ):
            raise ValueError("local embedding runtime returned a partial result")
        return vectors

    def close(self) -> None:
        self._model = None

    @staticmethod
    def _tree_fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        files = (
            [path]
            if path.is_file()
            else sorted(item for item in path.rglob("*") if item.is_file())
        )
        for item in files:
            digest.update(
                str(item.relative_to(path) if path.is_dir() else item.name).encode()
            )
            with item.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"


__all__ = ["SentenceTransformerVectorizer"]
