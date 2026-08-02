"""Tests for atomic vector projection persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgermind_local.projections import VectorProjectionStore


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def test_vector_store_rebuild_roundtrip_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(
        root,
        projection_version=2,
        model_fingerprint="fp-v1",
        model_name="vector-model.gguf",
        model_dimension=2,
        last_event_id="evt-1",
    )
    store.rebuild(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
    assert store.ids == ("a", "b")
    assert store.vectors == ((1.0, 2.0), (3.0, 4.0))
    assert store.dimension == 2
    assert store.document_count == 2
    assert not store.dirty

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a", "b")
    assert reloaded.vectors == ((1.0, 2.0), (3.0, 4.0))
    assert reloaded.dimension == 2
    assert reloaded.manifest["projection_version"] == 2
    assert reloaded.manifest["dimension"] == 2
    assert reloaded.manifest["model_fingerprint"] == "fp-v1"
    assert reloaded.manifest["model_name"] == "vector-model.gguf"
    assert reloaded.manifest["last_event_id"] == "evt-1"


def test_vector_store_upsert_updates_existing_ids(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
    store.upsert(["a", "c"], [[9.0, 9.0], [5.0, 6.0]])

    assert store.ids == ("a", "b", "c")
    assert store.vectors == ((9.0, 9.0), (3.0, 4.0), (5.0, 6.0))
    store.close()

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a", "b", "c")
    assert reloaded.vectors == ((9.0, 9.0), (3.0, 4.0), (5.0, 6.0))


def test_vector_store_remove_persists_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a", "b", "c"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    store.remove(["b"])
    assert store.ids == ("a", "c")
    store.close()

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a", "c")
    assert reloaded.vectors == ((1.0, 2.0), (5.0, 6.0))


def test_vector_store_close_flushes_dirty_state(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a"], [[1.0, 2.0]])
    store.upsert(["b"], [[3.0, 4.0]])
    assert store.dirty

    store.close()
    assert not store.dirty
    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a", "b")


def test_vector_store_rebuild_rejects_duplicate_ids(tmp_path: Path) -> None:
    store = VectorProjectionStore(tmp_path / "vector", model_dimension=2)

    with pytest.raises(ValueError, match="duplicate knowledge id in batch"):
        store.rebuild(["a", "a"], [[1.0, 2.0], [3.0, 4.0]])


def test_vector_store_rebuild_rejects_length_mismatch(tmp_path: Path) -> None:
    store = VectorProjectionStore(tmp_path / "vector", model_dimension=2)

    with pytest.raises(ValueError, match="knowledge_id and vector length mismatch"):
        store.rebuild(["a", "b"], [[1.0, 2.0]])


def test_vector_store_failure_after_vectors_write_keeps_previous_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a"], [[1.0, 2.0]])

    def fail(_: Path, __: list[list[float]]) -> None:
        raise RuntimeError("vector write failed")

    monkeypatch.setattr(store, "_write_vectors", fail)

    with pytest.raises(RuntimeError, match="vector write failed"):
        store.rebuild(["a"], [[9.0, 9.0]])

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a",)
    assert reloaded.vectors == ((1.0, 2.0),)
    assert reloaded.manifest["document_count"] == 1


def test_vector_store_failure_after_ids_write_keeps_previous_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a"], [[1.0, 2.0]])

    original_write_json = store._write_json
    written_ids_once = False

    def fail_on_ids(path: Path, value: object) -> None:
        nonlocal written_ids_once
        if path.name == "ids.json" and not written_ids_once:
            written_ids_once = True
            raise RuntimeError("ids write failed")
        original_write_json(path, value)

    monkeypatch.setattr(store, "_write_json", fail_on_ids)

    with pytest.raises(RuntimeError, match="ids write failed"):
        store.rebuild(["a"], [[3.0, 4.0]])

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a",)
    assert reloaded.vectors == ((1.0, 2.0),)


def test_vector_store_corrupted_json_fails_load(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    root.mkdir()
    _write_json(root / "ids.json", ["a", "b"])
    _write_json(root / "vectors.npy", [[1.0, 2.0], [3.0, 4.0]])
    (root / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="is corrupted"):
        VectorProjectionStore(root, model_dimension=2)


def test_vector_store_load_rejects_dimension_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a"], [[1.0, 2.0]])

    manifest = {
        "projection_version": 1,
        "model_fingerprint": None,
        "model_name": None,
        "dimension": 3,
        "document_count": 1,
        "last_event_id": None,
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    _write_json(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="dimension"):
        VectorProjectionStore(root, model_dimension=2)


def test_vector_store_load_rejects_duplicate_or_excess_ids(tmp_path: Path) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])

    _write_json(root / "ids.json", ["a", "a"])
    with pytest.raises(ValueError, match="contains duplicates"):
        VectorProjectionStore(root, model_dimension=2)

    _write_json(root / "ids.json", ["a", "b"])
    _write_json(root / "vectors.npy", [[1.0, 2.0]])
    with pytest.raises(ValueError, match="vector length mismatch"):
        VectorProjectionStore(root, model_dimension=2)


def test_vector_store_interrupted_rebuild_keeps_previous_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "vector"
    store = VectorProjectionStore(root, model_dimension=2)
    store.rebuild(["a"], [[1.0, 2.0]])

    monkeypatch.setattr(store, "_checkpoint", lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        store.rebuild(["a"], [[3.0, 4.0]])

    reloaded = VectorProjectionStore(root, model_dimension=2)
    assert reloaded.ids == ("a",)
    assert reloaded.vectors == ((1.0, 2.0),)
