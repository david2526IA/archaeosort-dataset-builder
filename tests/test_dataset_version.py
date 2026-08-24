from __future__ import annotations

import json
from pathlib import Path

from archaeosort_dataset_builder.versioning.dataset_version import (
    create_manifest,
    save_manifest,
    sha256_directory,
)


def _make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    (dataset / "train" / "natural_rock").mkdir(parents=True)
    (dataset / "train" / "natural_rock" / "img.jpg").write_bytes(b"rock-bytes")
    (dataset / "val" / "stone_artefact").mkdir(parents=True)
    (dataset / "val" / "stone_artefact" / "img.jpg").write_bytes(b"artefact-bytes")
    return dataset


def test_sha256_is_deterministic_for_same_content(tmp_path):
    dataset_a = _make_dataset(tmp_path / "a")
    dataset_b = _make_dataset(tmp_path / "b")

    assert sha256_directory(dataset_a) == sha256_directory(dataset_b)


def test_sha256_changes_when_content_changes(tmp_path):
    dataset = _make_dataset(tmp_path)
    before = sha256_directory(dataset)

    (dataset / "train" / "natural_rock" / "img.jpg").write_bytes(b"different-bytes")

    after = sha256_directory(dataset)
    assert before != after


def test_manifest_json_does_not_pollute_its_own_hash(tmp_path):
    dataset = _make_dataset(tmp_path)

    fingerprint_before_manifest = sha256_directory(dataset)

    save_manifest(dataset)
    fingerprint_after_first_manifest = create_manifest(dataset)["sha256"]

    # Writing manifest.json must not change the content fingerprint, and
    # rebuilding the manifest again (with manifest.json now already present
    # on disk) must produce the exact same sha256 as before it existed.
    assert fingerprint_before_manifest == fingerprint_after_first_manifest

    # Simulate a second build/manifest run on the same directory.
    save_manifest(dataset)
    fingerprint_after_second_manifest = create_manifest(dataset)["sha256"]

    assert fingerprint_after_first_manifest == fingerprint_after_second_manifest


def test_manifest_file_counts_exclude_manifest_itself(tmp_path):
    dataset = _make_dataset(tmp_path)

    manifest = create_manifest(dataset)
    assert manifest["files"] == 2  # the two synthetic images, not manifest.json

    save_manifest(dataset)
    manifest_after = create_manifest(dataset)
    assert manifest_after["files"] == 2


def test_save_manifest_writes_valid_json_and_returns_path(tmp_path):
    dataset = _make_dataset(tmp_path)

    manifest_path = save_manifest(dataset)

    assert manifest_path == dataset / "manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "sha256" in on_disk
    assert on_disk["files"] == 2


def test_build_bookkeeping_files_do_not_affect_fingerprint(tmp_path):
    # Regression test for a real bug found during the Fase F smoke test:
    # split_report.json and provenance.json are written by
    # builder.dataset_builder into the SAME output directory as the dataset
    # content, and provenance.json contains a "created" timestamp that
    # differs on every build. If either file were hashed, two builds of the
    # identical dataset content would get different manifest sha256 values.
    dataset = _make_dataset(tmp_path)
    fingerprint_before = sha256_directory(dataset)

    (dataset / "split_report.json").write_text(
        json.dumps({"seed": 42, "splits": {}}), encoding="utf-8"
    )
    (dataset / "provenance.json").write_text(
        json.dumps({"created": "2026-08-24T13:10:12.369019+00:00"}), encoding="utf-8"
    )

    fingerprint_with_bookkeeping_files = sha256_directory(dataset)
    assert fingerprint_before == fingerprint_with_bookkeeping_files

    manifest = create_manifest(dataset)
    assert manifest["files"] == 2  # only the two synthetic images
    assert manifest["sha256"] == fingerprint_before

    # A second "build" with a different provenance.json timestamp (as a
    # real rebuild would produce) must still yield the same fingerprint.
    (dataset / "provenance.json").write_text(
        json.dumps({"created": "2099-01-01T00:00:00+00:00"}), encoding="utf-8"
    )
    assert sha256_directory(dataset) == fingerprint_before


def test_sha256_directory_reads_in_chunks_for_large_file(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    # 3 MiB file, larger than the 1 MiB CHUNK_SIZE, to exercise the
    # chunked-read loop instead of a single read_bytes() call.
    large_file = dataset / "big.bin"
    large_file.write_bytes(b"x" * (3 * 1024 * 1024))

    # Should not raise and should be stable across two computations.
    assert sha256_directory(dataset) == sha256_directory(dataset)
