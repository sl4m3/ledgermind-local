"""Atomic vector projection store backed by filesystem files."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_numpy: Any = None
try:
    _numpy = __import__("numpy")
except ModuleNotFoundError:
    pass

np: Any = _numpy


_DEFAULT_TIME: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat()


class VectorProjectionStore:
    """Persisted vector space with atomic checkpoint semantics."""

    IDS_FILE = "ids.json"
    VECTORS_FILE = "vectors.npy"
    MANIFEST_FILE = "manifest.json"
    DEFAULT_PROJECTION_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        projection_version: int = DEFAULT_PROJECTION_VERSION,
        model_fingerprint: str | None = None,
        model_name: str | None = None,
        model_dimension: int | None = None,
        last_event_id: str | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

        self._projection_version = projection_version
        self._model_fingerprint = model_fingerprint
        self._model_name = model_name
        self._model_dimension = model_dimension
        self._last_event_id = last_event_id

        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._dirty = False
        self._load_if_exists()

        if self._model_dimension is not None and self._model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if self._model_dimension is not None and self._ids:
            self._ensure_dimension(self._vectors)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    @property
    def vectors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(item) for item in self._vectors)

    @property
    def document_count(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int | None:
        return self._model_dimension

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "projection_version": self._projection_version,
            "model_fingerprint": self._model_fingerprint,
            "model_name": self._model_name,
            "dimension": self._model_dimension,
            "document_count": len(self._ids),
            "last_event_id": self._last_event_id,
            "built_at": _DEFAULT_TIME(),
        }

    @property
    def dirty(self) -> bool:
        return self._dirty

    def upsert(
        self, knowledge_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        self._validate_lengths(knowledge_ids, vectors)
        if len(set(knowledge_ids)) != len(knowledge_ids):
            raise ValueError("duplicate knowledge id in batch")

        self._ensure_dimension(vectors)
        self._upsert_in_memory(knowledge_ids=knowledge_ids, vectors=vectors)
        self._dirty = True

    def remove(self, knowledge_ids: Sequence[str]) -> None:
        to_remove = {self._coerce_id(item) for item in knowledge_ids}
        if not to_remove:
            return

        kept_ids: list[str] = []
        kept_vectors: list[list[float]] = []
        changed = False
        for knowledge_id, vector in zip(self._ids, self._vectors):
            if knowledge_id in to_remove:
                changed = True
                continue
            kept_ids.append(knowledge_id)
            kept_vectors.append(vector)

        if changed:
            self._ids = kept_ids
            self._vectors = kept_vectors
            self._dirty = True

    def close(self) -> None:
        if self._dirty:
            self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        self._checkpoint()
        self._dirty = False

    def rebuild(
        self, knowledge_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        self._validate_lengths(knowledge_ids, vectors)
        self._ensure_dimension(vectors)

        ids = list(knowledge_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate knowledge id in batch")

        self._ids = ids
        self._vectors = [self._coerce_vector(item) for item in vectors]
        self._dirty = True
        self._checkpoint()
        self._dirty = False

    def _load_if_exists(self) -> None:
        ids_path = self._root / self.IDS_FILE
        vectors_path = self._root / self.VECTORS_FILE
        manifest_path = self._root / self.MANIFEST_FILE

        if (
            not ids_path.exists()
            and not vectors_path.exists()
            and not manifest_path.exists()
        ):
            return

        if (
            not ids_path.exists()
            or not vectors_path.exists()
            or not manifest_path.exists()
        ):
            raise ValueError("vector projection index is incomplete")

        manifest = self._load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json is malformed")  # noqa: TRY004 - persisted data contract

        stored_projection_version = self._coerce_int(manifest.get("projection_version"))
        if stored_projection_version is not None:
            self._projection_version = stored_projection_version

        ids = self._load_json(ids_path)
        if not isinstance(ids, list):
            raise ValueError("ids.json is malformed")  # noqa: TRY004 - persisted data contract

        loaded_ids = [self._coerce_id(item) for item in ids]
        if len(set(loaded_ids)) != len(loaded_ids):
            raise ValueError("ids.json contains duplicates")

        stored_dimension = self._coerce_int(manifest.get("dimension"))
        if stored_dimension is not None:
            if stored_dimension <= 0:
                raise ValueError("dimension must be positive")
            if self._model_dimension is None:
                self._model_dimension = stored_dimension
            elif self._model_dimension != stored_dimension:
                raise ValueError("vector dimension does not match store dimension")
        elif self._model_dimension is not None and self._model_dimension <= 0:
            raise ValueError("model_dimension must be positive")

        vectors = self._load_vectors(vectors_path)
        if len(vectors) != len(loaded_ids):
            raise ValueError("vector length mismatch")

        if manifest.get("model_fingerprint") is not None:
            self._model_fingerprint = str(manifest["model_fingerprint"])
        if manifest.get("model_name") is not None:
            self._model_name = str(manifest["model_name"])
        if manifest.get("last_event_id") is not None:
            self._last_event_id = str(manifest["last_event_id"])

        self._ensure_dimension(vectors)
        self._ids = [str(item) for item in ids]
        self._vectors = [self._coerce_vector(row) for row in vectors]

    def _checkpoint(self) -> None:
        manifest = self.manifest
        manifest["document_count"] = len(self._ids)

        self._persist(
            root=self._root, ids=self._ids, vectors=self._vectors, manifest=manifest
        )

    def _upsert_in_memory(
        self, knowledge_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        existing_index = {
            knowledge_id: idx for idx, knowledge_id in enumerate(self._ids)
        }
        for knowledge_id, vector in zip(knowledge_ids, vectors):
            normalized_id = str(knowledge_id)
            mapped = existing_index.get(normalized_id)
            normalized_vector = self._coerce_vector(vector)
            if mapped is None:
                existing_index[normalized_id] = len(self._ids)
                self._ids.append(normalized_id)
                self._vectors.append(normalized_vector)
            else:
                self._vectors[mapped] = normalized_vector

    def _validate_lengths(
        self,
        knowledge_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(knowledge_ids) != len(vectors):
            raise ValueError("knowledge_id and vector length mismatch")
        for knowledge_id in knowledge_ids:
            self._coerce_id(knowledge_id)

    def _coerce_id(self, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("knowledge_id must be a string")  # noqa: TRY004 - persisted data contract
        return value

    def _ensure_dimension(self, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        dimension = len(self._coerce_vector(vectors[0]))
        if dimension <= 0:
            raise ValueError("vector dimension must be positive")
        for vector in vectors:
            if len(self._coerce_vector(vector)) != dimension:
                raise ValueError("vectors must be consistent in dimension")
        if self._model_dimension is not None and dimension != self._model_dimension:
            raise ValueError("vector dimension does not match store dimension")
        self._model_dimension = dimension

    def _persist(
        self,
        *,
        root: Path,
        ids: list[str],
        vectors: list[list[float]],
        manifest: dict[str, Any],
    ) -> None:
        if root.exists() and not root.is_dir():
            raise RuntimeError("vector projection root is not a directory")

        tmp_parent = root.parent
        with tempfile.TemporaryDirectory(dir=tmp_parent) as temp_root:
            candidate = Path(temp_root) / root.name
            candidate.mkdir()
            ids_path = candidate / self.IDS_FILE
            vectors_path = candidate / self.VECTORS_FILE
            manifest_path = candidate / self.MANIFEST_FILE

            self._write_json(ids_path, ids)
            self._write_vectors(vectors_path, vectors)
            self._write_json(manifest_path, manifest)

            self._verify_candidate(candidate)

            backup: Path | None = None
            if root.exists():
                backup = Path(f"{root}.{os.getpid()}.old")
                os.replace(root, backup)
                try:
                    os.replace(candidate, root)
                except Exception:
                    os.replace(backup, root)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
            else:
                os.replace(candidate, root)

    def _verify_candidate(self, candidate: Path) -> None:
        ids = self._load_json(candidate / self.IDS_FILE)
        vectors = self._load_vectors(candidate / self.VECTORS_FILE)
        manifest = self._load_json(candidate / self.MANIFEST_FILE)

        if not isinstance(ids, list):
            raise ValueError("candidate ids is malformed")  # noqa: TRY004 - persisted data contract
        if not isinstance(manifest, dict):
            raise ValueError("candidate manifest is malformed")  # noqa: TRY004 - persisted data contract
        if not isinstance(vectors, list):
            raise ValueError("candidate vectors is malformed")  # noqa: TRY004 - persisted data contract
        if any(not isinstance(value, str) for value in ids):
            raise ValueError("candidate ids must be string values")
        if len(ids) != len(vectors):
            raise ValueError("candidate vector length mismatch")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids contain duplicates")

        manifest_dimension = manifest.get("dimension")
        if manifest_dimension is not None:
            model_dimension = self._coerce_int(manifest_dimension)
            if model_dimension is not None:
                for vector in vectors:
                    if len(self._coerce_vector(vector)) != model_dimension:
                        raise ValueError("candidate dimension mismatch")

    def _load_json(self, path: Path) -> dict[str, Any] | list[Any] | str:
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is corrupted") from exc
        if not isinstance(payload, (dict, list, str)):
            raise ValueError(f"{path.name} has unexpected content type")  # noqa: TRY004 - persisted data contract
        return payload

    def _write_json(self, path: Path, value: object) -> None:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        path.write_text(payload, encoding="utf-8")

    def _write_vectors(self, path: Path, vectors: list[list[float]]) -> None:
        if np is None:
            path.write_text(
                json.dumps(
                    vectors, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
                encoding="utf-8",
            )
            return

        if self._model_dimension is not None and not vectors:
            array = np.empty((0, self._model_dimension), dtype=np.float32)
        else:
            array = np.asarray(vectors, dtype=np.float32)
        with path.open("wb") as handle:
            np.save(handle, array)

    def _load_vectors(self, path: Path) -> list[list[float]]:
        if not path.exists():
            raise ValueError("vectors file is missing")
        if np is None:
            raw_text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError("vectors file is corrupted") from exc
            if not isinstance(payload, list):
                raise ValueError("vectors file is malformed")
            return [self._coerce_vector(row) for row in payload]

        raw_bytes = path.read_bytes()
        try:
            array = np.load(path, allow_pickle=False)
        except Exception:  # noqa: BLE001 - fallback supports JSON vector stores
            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                raise ValueError("vectors file is corrupted") from exc
            if not isinstance(payload, list):
                raise ValueError("vectors file is malformed")  # noqa: TRY004 - persisted data contract
            return [self._coerce_vector(row) for row in payload]

        vector_list = array.tolist()
        return [self._coerce_vector(row) for row in vector_list]

    def _coerce_vector(self, vector: Sequence[float] | Any) -> list[float]:
        if not isinstance(vector, Sequence) or isinstance(
            vector, (str, bytes, bytearray)
        ):
            raise ValueError("vector must be a sequence of numbers")  # noqa: TRY004 - persisted data contract
        coerced: list[float] = [float(value) for value in vector]
        return coerced

    def _coerce_int(self, value: object) -> int | None:
        if value is None:
            return None
        try:
            coerced = int(str(value))
        except (TypeError, ValueError):
            return None
        return coerced
