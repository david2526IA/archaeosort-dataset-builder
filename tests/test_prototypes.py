"""Tests for prototypes/prototypes.py. embed_image (torch-dependent, lives
in api.similarity) is monkeypatched so these run without torch/a real
DINOv2 model -- they exercise the prototype math (averaging, L2
normalization, cosine similarity, softmax) and the save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from archaeosort_dataset_builder.prototypes.prototypes import (
    PrototypeBundle,
    PrototypeError,
    build_prototypes,
    classify_by_prototype,
    load_prototypes,
    save_prototypes,
)


@pytest.fixture
def fake_embed_image(monkeypatch):
    """Map a small set of sentinel 'images' (plain strings, standing in
    for PIL Images) to fixed 3D embedding vectors, so prototype math is
    deterministic and torch-free."""
    vectors = {
        "rock_a": np.array([1.0, 0.0, 0.0]),
        "rock_b": np.array([0.9, 0.1, 0.0]),
        "bone_a": np.array([0.0, 1.0, 0.0]),
        "bone_b": np.array([0.0, 0.9, 0.1]),
        "query_like_rock": np.array([0.95, 0.05, 0.0]),
        "query_like_bone": np.array([0.05, 0.95, 0.0]),
    }

    def _fake_embed_image(image):
        return vectors[image].astype("float32")

    monkeypatch.setattr(
        "archaeosort_dataset_builder.api.similarity.embed_image", _fake_embed_image
    )
    return vectors


def test_build_prototypes_averages_and_normalizes(fake_embed_image):
    bundle = build_prototypes({"rock": ["rock_a", "rock_b"], "bone": ["bone_a", "bone_b"]})
    assert bundle.labels == ["rock", "bone"]
    assert bundle.vectors.shape == (2, 3)
    # Each prototype row must be unit-length (L2-normalized).
    norms = np.linalg.norm(bundle.vectors, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


def test_build_prototypes_rejects_empty_label_list(fake_embed_image):
    with pytest.raises(PrototypeError):
        build_prototypes({"rock": ["rock_a"], "bone": []})


def test_build_prototypes_rejects_no_categories():
    with pytest.raises(PrototypeError):
        build_prototypes({})


def test_classify_by_prototype_picks_nearest_class(fake_embed_image):
    bundle = build_prototypes({"rock": ["rock_a", "rock_b"], "bone": ["bone_a", "bone_b"]})

    rock_result = classify_by_prototype(bundle, "query_like_rock")
    assert rock_result["predicted_class"] == "rock"
    assert set(rock_result["probabilities"]) == {"rock", "bone"}
    assert rock_result["confidence"] == pytest.approx(max(rock_result["probabilities"].values()))

    bone_result = classify_by_prototype(bundle, "query_like_bone")
    assert bone_result["predicted_class"] == "bone"


def test_classify_by_prototype_probabilities_sum_to_one(fake_embed_image):
    bundle = build_prototypes({"rock": ["rock_a"], "bone": ["bone_a"]})
    result = classify_by_prototype(bundle, "query_like_rock")
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_save_and_load_prototypes_round_trip(tmp_path: Path, fake_embed_image):
    bundle = build_prototypes({"rock": ["rock_a", "rock_b"], "bone": ["bone_a", "bone_b"]})
    path = tmp_path / "prototypes.npz"

    save_prototypes(bundle, path)
    loaded = load_prototypes(path)

    assert loaded.labels == bundle.labels
    np.testing.assert_allclose(loaded.vectors, bundle.vectors)


def test_load_prototypes_missing_file_raises(tmp_path: Path):
    with pytest.raises(PrototypeError):
        load_prototypes(tmp_path / "does_not_exist.npz")


def test_load_prototypes_corrupt_file_raises(tmp_path: Path):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a valid npz file")
    with pytest.raises(PrototypeError):
        load_prototypes(path)


def test_prototype_bundle_is_a_plain_dataclass():
    bundle = PrototypeBundle(labels=["a", "b"], vectors=np.zeros((2, 3)))
    assert bundle.labels == ["a", "b"]
    assert bundle.vectors.shape == (2, 3)
