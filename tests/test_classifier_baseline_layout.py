"""Tests for the torch-free part of classifier/baseline.py.

The rest of the module (build_dataloaders, build_model, train_baseline)
needs torch/torchvision/scikit-learn, which are not available in this
lightweight verification sandbox (see the session notes for why) — those
must be exercised with the real `uv run --no-sync pytest -q` on the
project's actual environment, where torch is already a dependency. This
file only covers validate_split_layout, which has no heavy imports and can
be verified here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archaeosort_dataset_builder.classifier.baseline import (
    TrainingError,
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
