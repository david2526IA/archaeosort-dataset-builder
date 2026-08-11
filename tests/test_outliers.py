import json

import numpy as np
import pytest

from archaeosort_dataset_builder.embeddings import outliers


def configure_files(tmp_path, monkeypatch, vectors, paths):
    outputs = tmp_path / "outputs"
    reports = tmp_path / "reports"
    embedding_dir = outputs / "embeddings"
    embedding_dir.mkdir(parents=True)

    np.save(
        embedding_dir / "dinov2_embeddings.npy",
        np.asarray(vectors, dtype=np.float32),
    )

    (embedding_dir / "dinov2_paths.json").write_text(
        json.dumps(paths),
        encoding="utf8",
    )

    monkeypatch.setattr(outliers.settings, "outputs", outputs)
    monkeypatch.setattr(outliers.settings, "reports", reports)

    return reports


def test_get_label():
    assert outliers.get_label("dataset/natural_rock/image.jpg") == "natural_rock"


def test_detect_outliers_creates_report(tmp_path, monkeypatch):
    reports = configure_files(
        tmp_path,
        monkeypatch,
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        [
            "dataset/rock/a.jpg",
            "dataset/rock/b.jpg",
            "dataset/artefact/c.jpg",
            "dataset/artefact/d.jpg",
        ],
    )

    report = outliers.detect_outliers(k=1, percentile=50)

    assert report["images"] == 4
    assert report["neighbors"] == 1
    assert (reports / "outliers.json").exists()


def test_single_image_class_is_skipped(tmp_path, monkeypatch):
    configure_files(
        tmp_path,
        monkeypatch,
        [[1.0, 0.0]],
        ["dataset/single/image.jpg"],
    )

    report = outliers.detect_outliers()

    assert report["outliers"] == 0
    assert report["skipped_classes"] == ["single"]


def test_embedding_path_count_must_match(tmp_path, monkeypatch):
    configure_files(
        tmp_path,
        monkeypatch,
        [[1.0, 0.0], [0.0, 1.0]],
        ["dataset/rock/a.jpg"],
    )

    with pytest.raises(ValueError, match="does not match"):
        outliers.detect_outliers()


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must"):
        outliers.detect_outliers(k=k)


@pytest.mark.parametrize("percentile", [-1, 101])
def test_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentile"):
        outliers.detect_outliers(percentile=percentile)
