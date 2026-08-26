"""Tests for api/similarity.py's index-loading and result-shaping logic.
`load_similarity_index` only needs faiss/numpy/json (no torch), so it's
exercised here against small synthetic FAISS indexes. `embed_image` needs
torch + the DINOv2 model and is only exercised structurally elsewhere (via
the API's monkeypatched tests in test_api_app.py); real embedding quality is
verified on the project's real environment, not here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from archaeosort_dataset_builder.api.similarity import (
    SimilarityError,
    load_similarity_index,
    search_similar,
)


def _write_index(tmp_path, vectors, paths):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    faiss.write_index(index, str(tmp_path / "dinov2.faiss"))
    (tmp_path / "dinov2_paths.json").write_text(json.dumps(paths), encoding="utf-8")


def test_load_similarity_index_missing_index_file(tmp_path):
    (tmp_path / "dinov2_paths.json").write_text(json.dumps(["a.jpg"]), encoding="utf-8")
    with pytest.raises(SimilarityError, match="FAISS index not found"):
        load_similarity_index(tmp_path)


def test_load_similarity_index_missing_paths_file(tmp_path):
    import faiss

    index = faiss.IndexFlatIP(4)
    index.add(np.zeros((1, 4), dtype="float32"))
    faiss.write_index(index, str(tmp_path / "dinov2.faiss"))
    with pytest.raises(SimilarityError, match="Paths file not found"):
        load_similarity_index(tmp_path)


def test_load_similarity_index_mismatched_counts(tmp_path):
    _write_index(tmp_path, np.random.rand(3, 4), ["a.jpg", "b.jpg"])
    with pytest.raises(SimilarityError, match="mismatch"):
        load_similarity_index(tmp_path)


def test_load_similarity_index_empty_paths(tmp_path):
    import faiss

    index = faiss.IndexFlatIP(4)
    faiss.write_index(index, str(tmp_path / "dinov2.faiss"))
    (tmp_path / "dinov2_paths.json").write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(SimilarityError, match="non-empty"):
        load_similarity_index(tmp_path)


def test_load_and_search_similarity_index(tmp_path, monkeypatch):
    vectors = np.eye(4, dtype="float32")  # 4 orthonormal vectors -> exact self-match
    paths = [f"/data/img_{i}.jpg" for i in range(4)]
    _write_index(tmp_path, vectors, paths)

    bundle = load_similarity_index(tmp_path)
    assert bundle.paths == paths

    import archaeosort_dataset_builder.api.similarity as similarity_module

    monkeypatch.setattr(similarity_module, "embed_image", lambda image: vectors[2])

    results = search_similar(bundle, image=object(), k=2)
    assert results[0]["path_id"] == 2
    assert results[0]["filename"] == "img_2.jpg"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["rank"] == 1
    assert len(results) == 2


def test_search_similar_caps_k_to_index_size(tmp_path, monkeypatch):
    vectors = np.eye(3, dtype="float32")
    paths = [f"img_{i}.jpg" for i in range(3)]
    _write_index(tmp_path, vectors, paths)
    bundle = load_similarity_index(tmp_path)

    import archaeosort_dataset_builder.api.similarity as similarity_module

    monkeypatch.setattr(similarity_module, "embed_image", lambda image: vectors[0])

    results = search_similar(bundle, image=object(), k=100)
    assert len(results) == 3
