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


def test_build_reference_dataset_reuses_existing_when_at_target(tmp_path: Path):
    output_dir = tmp_path / "references"
    label_dir = output_dir / "bone_artefact"
    label_dir.mkdir(parents=True)
    for i in range(6):
        (label_dir / f"{i}.jpg").write_bytes(b"already here")

    # A source that would explode if actually called (no network mocking
    # set up) -- proves reuse short-circuits before touching the network,
    # because existing (6) already meets per_class (6).
    manifest = build_reference_dataset(
        output_dir,
        sources={"bone_artefact": CategorySource(kind="met", met_queries=["should not be queried"])},
        per_class=6,
    )

    assert manifest["categories"]["bone_artefact"]["reused_existing"] is True
    assert manifest["categories"]["bone_artefact"]["count"] == 6


def test_build_reference_dataset_tops_up_when_below_target(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "references"
    label_dir = output_dir / "bone_artefact"
    label_dir.mkdir(parents=True)
    for i in range(3):
        (label_dir / f"{i:03d}.jpg").write_bytes(b"already here")

    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "collectionapi.metmuseum.org/public/collection/v1/search" in url:
            return _FakeResponse(json_data={"objectIDs": [1, 2, 3, 4, 5]})
        if "collectionapi.metmuseum.org/public/collection/v1/objects" in url:
            return _FakeResponse(
                json_data={"isPublicDomain": True, "primaryImageSmall": "https://example.com/img.jpg"}
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    manifest = build_reference_dataset(
        output_dir,
        sources={"bone_artefact": CategorySource(kind="met", met_queries=["carved bone"])},
        per_class=8,
    )

    # 3 pre-existing + 5 newly fetched = 8 total; the 3 original files
    # must survive untouched (top-up numbers new files from index 3).
    assert manifest["categories"]["bone_artefact"]["count"] == 8
    assert len(list((output_dir / "bone_artefact").glob("*.jpg"))) == 8
    for i in range(3):
        assert (output_dir / "bone_artefact" / f"{i:03d}.jpg").read_bytes() == b"already here"


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


def test_build_reference_dataset_inaturalist_source(tmp_path: Path, monkeypatch):
    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "api.inaturalist.org/v1/observations" in url:
            page = (params or {}).get("page", 1)
            if page > 1:
                return _FakeResponse(json_data={"results": []})
            return _FakeResponse(
                json_data={
                    "results": [
                        {"photos": [{"id": i, "url": f"https://example.com/photos/{i}/square.jpg"}]}
                        for i in range(1, 6)
                    ]
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"natural_bone": CategorySource(kind="inaturalist", inaturalist_queries=["animal bone"])},
        per_class=5,
    )

    assert manifest["categories"]["natural_bone"]["count"] == 5
    assert manifest["categories"]["natural_bone"]["source"] == "inaturalist"


def test_build_reference_dataset_multi_source_combines_met_and_commons(tmp_path: Path, monkeypatch):
    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "collectionapi.metmuseum.org/public/collection/v1/search" in url:
            return _FakeResponse(json_data={"objectIDs": [1, 2, 3]})
        if "collectionapi.metmuseum.org/public/collection/v1/objects" in url:
            return _FakeResponse(
                json_data={"isPublicDomain": True, "primaryImageSmall": "https://example.com/img.jpg"}
            )
        if "commons.wikimedia.org" in url and (params or {}).get("list") == "categorymembers":
            return _FakeResponse(
                json_data={"query": {"categorymembers": [{"title": "File:A.jpg"}, {"title": "File:B.jpg"}]}}
            )
        if "commons.wikimedia.org" in url and (params or {}).get("prop") == "imageinfo":
            return _FakeResponse(
                json_data={
                    "query": {
                        "pages": {
                            "1": {"imageinfo": [{"thumburl": "https://example.com/a.jpg"}]},
                            "2": {"imageinfo": [{"thumburl": "https://example.com/b.jpg"}]},
                        }
                    }
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={
            "wood_artefact": CategorySource(
                kind="multi",
                met_queries=["wood carving"],
                commons_categories=["Category:Wood_carvings"],
            )
        },
        per_class=5,
    )

    assert manifest["categories"]["wood_artefact"]["count"] == 5
    assert "met_museum" in manifest["categories"]["wood_artefact"]["source"]
    assert "wikimedia_commons" in manifest["categories"]["wood_artefact"]["source"]


def test_build_reference_dataset_gbif_source(tmp_path: Path, monkeypatch):
    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "api.gbif.org/v1/occurrence/search" in url:
            offset = (params or {}).get("offset", 0)
            if offset > 0:
                return _FakeResponse(json_data={"results": []})
            return _FakeResponse(
                json_data={
                    "results": [
                        {"key": i, "media": [{"identifier": f"https://example.com/gbif/{i}.jpg"}]}
                        for i in range(1, 6)
                    ]
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"natural_bone": CategorySource(kind="gbif", gbif_queries=["skeleton"])},
        per_class=5,
    )

    assert manifest["categories"]["natural_bone"]["count"] == 5
    assert manifest["categories"]["natural_bone"]["source"] == "gbif"


def test_build_reference_dataset_idigbio_source(tmp_path: Path, monkeypatch):
    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "search.idigbio.org/v2/search/media" in url:
            return _FakeResponse(
                json_data={
                    "items": [
                        {"data": {"ac:accessURI": f"https://example.com/idigbio/{i}.jpg"}}
                        for i in range(1, 6)
                    ]
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"natural_bone": CategorySource(kind="idigbio", idigbio_queries=["skeleton"])},
        per_class=5,
    )

    assert manifest["categories"]["natural_bone"]["count"] == 5
    assert manifest["categories"]["natural_bone"]["source"] == "idigbio"


def test_build_reference_dataset_smithsonian_source_skipped_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SMITHSONIAN_API_KEY", raising=False)

    def fake_get_all(url, params=None, headers=None, timeout=None):
        raise AssertionError("should not call the network when SMITHSONIAN_API_KEY is unset")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    with pytest.raises(ReferenceDatasetError):
        build_reference_dataset(
            output_dir,
            sources={"bone_artefact": CategorySource(kind="smithsonian", smithsonian_queries=["bone tool"])},
            per_class=5,
        )


def test_build_reference_dataset_smithsonian_source_with_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SMITHSONIAN_API_KEY", "fake-key")

    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "api.si.edu/openaccess/api/v1.0/search" in url:
            start = (params or {}).get("start", 0)
            if start > 0:
                return _FakeResponse(json_data={"response": {"rows": []}})
            return _FakeResponse(
                json_data={
                    "response": {
                        "rows": [
                            {
                                "id": f"si-{i}",
                                "content": {
                                    "descriptiveNonRepeating": {
                                        "online_media": {
                                            "media": [
                                                {
                                                    "type": "Images",
                                                    "content": f"https://example.com/si/{i}.jpg",
                                                }
                                            ]
                                        }
                                    }
                                },
                            }
                            for i in range(1, 6)
                        ]
                    }
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"bone_artefact": CategorySource(kind="smithsonian", smithsonian_queries=["bone tool"])},
        per_class=5,
    )

    assert manifest["categories"]["bone_artefact"]["count"] == 5
    assert manifest["categories"]["bone_artefact"]["source"] == "smithsonian"


def test_build_reference_dataset_europeana_source_with_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EUROPEANA_API_KEY", "fake-key")

    def fake_get_all(url, params=None, headers=None, timeout=None):
        if "api.europeana.eu/record/v2/search.json" in url:
            start = (params or {}).get("start", 1)
            if start > 1:
                return _FakeResponse(json_data={"items": []})
            return _FakeResponse(
                json_data={
                    "items": [{"edmPreview": f"https://example.com/eu/{i}.jpg"} for i in range(1, 6)]
                }
            )
        return _FakeResponse(content=b"fake-image-bytes")

    monkeypatch.setattr("requests.get", fake_get_all)

    output_dir = tmp_path / "references"
    manifest = build_reference_dataset(
        output_dir,
        sources={"wood_artefact": CategorySource(kind="europeana", europeana_queries=["wood carving"])},
        per_class=5,
    )

    assert manifest["categories"]["wood_artefact"]["count"] == 5
    assert manifest["categories"]["wood_artefact"]["source"] == "europeana"
