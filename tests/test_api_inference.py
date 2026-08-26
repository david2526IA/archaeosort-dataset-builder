"""Torch-free tests for api/inference.py's class-resolution helpers.

`load_model_bundle`/`predict_image` need torch and are only exercised
structurally (import + signature checks) here; real verification happens on
the user's GPU machine, same convention as the rest of the classifier code.
"""

from __future__ import annotations

import json

import pytest

from archaeosort_dataset_builder.api.inference import (
    InferenceError,
    _read_classes_from_evaluation,
    resolve_classes,
)


def test_read_classes_from_evaluation_missing_file(tmp_path):
    model_path = tmp_path / "best_model.pt"
    assert _read_classes_from_evaluation(model_path) is None


def test_read_classes_from_evaluation_malformed_json(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text("{not valid json", encoding="utf-8")
    assert _read_classes_from_evaluation(model_path) is None


def test_read_classes_from_evaluation_missing_classes_field(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text(json.dumps({"test_accuracy": 0.95}), encoding="utf-8")
    assert _read_classes_from_evaluation(model_path) is None


def test_read_classes_from_evaluation_empty_classes_list(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text(json.dumps({"classes": []}), encoding="utf-8")
    assert _read_classes_from_evaluation(model_path) is None


def test_read_classes_from_evaluation_valid(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"classes": ["natural_rock", "stone_artefact"]}), encoding="utf-8"
    )
    assert _read_classes_from_evaluation(model_path) == ["natural_rock", "stone_artefact"]


def test_resolve_classes_explicit_override_takes_precedence(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"classes": ["from_evaluation_json"]}), encoding="utf-8"
    )
    assert resolve_classes(model_path, ["explicit_class"]) == ["explicit_class"]


def test_resolve_classes_falls_back_to_evaluation_json(tmp_path):
    model_path = tmp_path / "best_model.pt"
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"classes": ["natural_rock", "stone_artefact"]}), encoding="utf-8"
    )
    assert resolve_classes(model_path, None) == ["natural_rock", "stone_artefact"]


def test_resolve_classes_raises_when_nothing_available(tmp_path):
    model_path = tmp_path / "best_model.pt"
    with pytest.raises(InferenceError):
        resolve_classes(model_path, None)
