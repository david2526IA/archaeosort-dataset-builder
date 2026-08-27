"""Tests for the torch-free part of classifier/baseline.py.

The rest of the module (build_dataloaders, build_model, train_baseline)
needs torch/torchvision/scikit-learn, which are not available in this
lightweight verification sandbox (see the session notes for why) — those
must be exercised with the real `uv run --no-sync pytest -q` on the
project's actual environment, where torch is already a dependency. This
file covers validate_split_layout and compute_class_weights, neither of
which has heavy imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archaeosort_dataset_builder.classifier.baseline import (
    TrainingError,
    compute_class_weights,
    validate_split_layout,
)


def _make_split_dataset(root: Path, splits=("train", "val", "test")) -> Path:
    dataset = root / "dataset"
    for split in splits:
        (dataset / split / "natural_rock").mkdir(parents=True, exist_ok=True)
    return dataset


def test_validate_split_layout_accepts_all_three_splits(tmp_path):
    dataset = _make_split_dataset(tmp_path)
    result = validate_split_layout(dataset)
    assert result == dataset.resolve()


def test_validate_split_layout_rejects_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_split_layout(tmp_path / "does-not-exist")


def test_validate_split_layout_rejects_missing_split(tmp_path):
    dataset = _make_split_dataset(tmp_path, splits=("train", "val"))  # no test/
    with pytest.raises(TrainingError):
        validate_split_layout(dataset)


def test_validate_split_layout_rejects_a_source_class_folder_dataset(tmp_path):
    # A common mistake: pointing train-baseline at the *source* dataset
    # (class folders directly, the input to `archaeosort build`) instead of
    # build()'s train/val/test output. Should fail clearly, not silently
    # train on garbage.
    source_dataset = tmp_path / "source"
    (source_dataset / "natural_rock").mkdir(parents=True)
    (source_dataset / "stone_artefact").mkdir(parents=True)

    with pytest.raises(TrainingError):
        validate_split_layout(source_dataset)


def test_compute_class_weights_balanced_classes_get_equal_weight():
    # 10 samples per class, 2 classes -> each weight = 20 / (2 * 10) = 1.0
    targets = [0] * 10 + [1] * 10
    weights = compute_class_weights(targets, num_classes=2)
    assert weights == pytest.approx([1.0, 1.0])


def test_compute_class_weights_rare_class_gets_higher_weight():
    # class 0: 100 samples, class 1: 10 samples -> class 1 should weigh 10x more
    targets = [0] * 100 + [1] * 10
    weights = compute_class_weights(targets, num_classes=2)
    assert weights[1] == pytest.approx(weights[0] * 10)


def test_compute_class_weights_matches_the_robust_dataset_imbalance_shape():
    # Mirrors the real robust_v1 train split shape: one big class (stone),
    # one medium (natural_rock), four small (bone/wood) classes -- the
    # small ones must end up with substantially larger weights.
    targets = [0] * 2716 + [1] * 1006 + [2] * 175 + [3] * 175 + [4] * 175 + [5] * 175
    weights = compute_class_weights(targets, num_classes=6)
    assert weights[2] > weights[1] > weights[0]
    assert weights[2] == pytest.approx(weights[3])


def test_compute_class_weights_absent_class_gets_zero_weight():
    targets = [0] * 5  # class 1 never appears
    weights = compute_class_weights(targets, num_classes=2)
    assert weights[1] == 0.0


def test_compute_class_weights_rejects_non_positive_num_classes():
    with pytest.raises(ValueError):
        compute_class_weights([0, 1], num_classes=0)
