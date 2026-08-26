"""Tests for prototypes/reference_dataset.py. All network calls go through
`requests.get`, which is monkeypatched here -- no real HTTP requests are
made, so these run offline and deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archaeosort_dataset_builder.prototypes.reference_dataset import (
    CategorySource,
    ReferenceDatasetError,
    _sample_local_images,
    build_reference_dataset,
)


class _FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json_data = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_sample_local_images_filters_by_extension_and_respects_limit(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "c.txt").write_text("not an image")
    (tmp_path / "sub").mkdir()

    sampled = _sample_local_images(tmp_path, limit=1)
    assert len(sampled) == 1
    assert sampled[0].suffix in {".jpg", ".png"}


def test_build_reference_dataset_local_source_copies_files(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for i in range(6):
        (source_dir / f"img_{i}.jpg").write_bytes(f"fake-{i}".encode())

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"natural_rock": CategorySource(kind="local", local_dir=source_dir)},
        per_class=5,
    )

    assert manifest["categories"]["natural_rock"]["count"] == 5
    copied = list((output_dir / "natural_rock").glob("*.jpg"))
    assert len(copied) == 5


def test_build_reference_dataset_reuses_existing_by_default(tmp_path: Path):
    output_dir = tmp_path / "references"
    label_dir = output_dir / "bone_artefact"
    label_dir.mkdir(parents=True)
    for i in range(6):
        (label_dir / f"{i}.jpg").write_bytes(b"already here")

    # A source that would explode if actually called (no network mocking
    # set up) -- proves reuse short-circuits before touching the network.
    manifest = build_reference_dataset(
        output_dir,
        sources={"bone_artefact": CategorySource(kind="met", met_queries=["should not be queried"])},
        per_class=20,
    )

    assert manifest["categories"]["bone_artefact"]["reused_existing"] is True
    assert manifest["categories"]["bone_artefact"]["count"] == 6


def test_build_reference_dataset_met_source(tmp_path: Path, monkeypatch):
    # _download_bytes and _http_get_json both call requests.get -- one
    # fake handles both, since _FakeResponse supports .json() and .content.
    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "collectionapi.metmuseum.org/public/collection/v1/search" in url:
            return _FakeResponse(json_data={"objectIDs": [1, 2, 3, 4, 5, 6]})
        if "collectionapi.metmuseum.org/public/collection/v1/objects" in url:
            return _FakeResponse(
                json_data={"isPublicDomain": True, "primaryImageSmall": "https://example.com/img.jpg"}
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"bone_artefact": CategorySource(kind="met", met_queries=["carved bone"])},
        per_class=5,
    )

    assert manifest["categories"]["bone_artefact"]["count"] == 5
    assert len(list((output_dir / "bone_artefact").glob("*.jpg"))) == 5


def test_build_reference_dataset_raises_when_too_few_images(tmp_path: Path, monkeypatch):
    def fake_get_empty(url, params=None, headers=None, timeout=None):
        return _FakeResponse(json_data={"objectIDs": []})

    monkeypatch.setattr("requests.get", fake_get_empty)

    output_dir = tmp_path / "references"
    with pytest.raises(ReferenceDatasetError):
        build_reference_dataset(
            output_dir,
            sources={"bone_artefact": CategorySource(kind="met", met_queries=["carved bone"])},
            per_class=5,
        )


def test_build_reference_dataset_unknown_source_kind_raises(tmp_path: Path):
    with pytest.raises(ReferenceDatasetError):
        build_reference_dataset(
            tmp_path / "references",
            sources={"weird": CategorySource(kind="not-a-real-kind")},
            per_class=5,
        )


def test_build_reference_dataset_local_missing_dir_raises(tmp_path: Path):
    with pytest.raises(ReferenceDatasetError):
        build_reference_dataset(
            tmp_path / "references",
            sources={"natural_rock": CategorySource(kind="local", local_dir=None)},
            per_class=5,
        )
