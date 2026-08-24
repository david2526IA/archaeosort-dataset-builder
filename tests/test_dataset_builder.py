from __future__ import annotations

import json
from pathlib import Path

import pytest

from archaeosort_dataset_builder.builder.dataset_builder import (
    BuildError,
    build_dataset,
)

# Counts chosen so that with the default 0.70/0.15/0.15 ratios every
# split ends up with at least one image per class (int() truncation means
# very small counts can leave a split at 0 images for a class, in which
# case build_split_dataset legitimately does not create that directory —
# that's existing splitter behavior, not something this test should hide).
CLASSES = {
    "natural_rock": 20,
    "stone_artefact": 14,
}


def _make_source_dataset(root: Path) -> Path:
    """Create a tiny synthetic class-folder dataset (empty placeholder files,
    the builder/splitter only care about path + extension, not real image
    bytes)."""
    source = root / "source"
    for class_name, count in CLASSES.items():
        class_dir = source / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (class_dir / f"img_{index:02d}.jpg").write_bytes(b"fake-image-bytes")
    return source


# 1. ratios sum to 1 (valid case does not raise)
def test_ratios_summing_to_one_accepted(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert result["output_dir"].exists()


# 2. negative ratio rejected
def test_negative_ratio_rejected(tmp_path):
    source = _make_source_dataset(tmp_path)
    with pytest.raises(ValueError):
        build_dataset(source, tmp_path / "out", train_ratio=1.1, val_ratio=0.0, test_ratio=-0.1)


# 3. missing source rejected
def test_missing_source_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "does-not-exist", tmp_path / "out")


# 4. empty/unclassified source rejected
def test_empty_source_rejected(tmp_path):
    empty_source = tmp_path / "empty-source"
    empty_source.mkdir()
    with pytest.raises(ValueError):
        build_dataset(empty_source, tmp_path / "out")


# 5. non-empty output without force rejected
def test_non_empty_output_without_force_rejected(tmp_path):
    source = _make_source_dataset(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "leftover.txt").write_text("old build")

    with pytest.raises(FileExistsError):
        build_dataset(source, output, force=False)


# 6. --force rebuild works
def test_force_rebuilds_existing_output(tmp_path):
    source = _make_source_dataset(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "leftover.txt").write_text("old build")

    result = build_dataset(source, output, force=True)

    assert not (output / "leftover.txt").exists()
    assert result["split_report_path"].exists()


# 7. same seed is deterministic
def test_same_seed_is_deterministic(tmp_path):
    source = _make_source_dataset(tmp_path)

    result_a = build_dataset(source, tmp_path / "out_a", seed=42, copy_files=True)
    result_b = build_dataset(source, tmp_path / "out_b", seed=42, copy_files=True)

    assert result_a["split_report"]["splits"] == result_b["split_report"]["splits"]


# 8. train/val/test generated
def test_train_val_test_dirs_generated(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out")

    for split in ("train", "val", "test"):
        assert (result["output_dir"] / split).is_dir()


# 9. class directories preserved
def test_class_directories_preserved(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", copy_files=True)

    for split in ("train", "val", "test"):
        split_dir = result["output_dir"] / split
        present_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        assert present_classes <= set(CLASSES)


# 10. total output images == source
def test_total_output_images_matches_source(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", copy_files=True)

    total_in_output = sum(1 for p in result["output_dir"].rglob("*.jpg"))
    assert total_in_output == sum(CLASSES.values())
    assert result["split_report"]["total_images"] == sum(CLASSES.values())


# 11. split_report.json exists and correct
def test_split_report_file_is_written_and_matches(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", copy_files=True)

    on_disk = json.loads(result["split_report_path"].read_text(encoding="utf-8"))
    assert on_disk == result["split_report"]


# 12. provenance.json exists and records seed/ratios/classes
def test_provenance_file_records_key_fields(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(
        source,
        tmp_path / "out",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
        copy_files=True,
    )

    on_disk = json.loads(result["provenance_path"].read_text(encoding="utf-8"))
    assert on_disk["seed"] == 7
    assert on_disk["ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert on_disk["classes"] == sorted(CLASSES)
    assert on_disk["source_image_count"] == sum(CLASSES.values())
    assert on_disk["mode"] == "copy"


# 13. manifest.json exists
def test_manifest_file_is_written(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", copy_files=True)

    assert result["manifest_path"].exists()
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert "sha256" in manifest


# 14. input == output rejected
def test_input_equal_to_output_rejected(tmp_path):
    source = _make_source_dataset(tmp_path)
    with pytest.raises(BuildError):
        build_dataset(source, source)


# 15. copy mode actually copies (source files untouched, real regular files in output)
def test_copy_mode_produces_independent_files(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out", copy_files=True)

    any_output_file = next(result["output_dir"].rglob("*.jpg"))
    assert not any_output_file.is_symlink()
    # source dataset must be untouched
    assert sum(1 for _ in source.rglob("*.jpg")) == sum(CLASSES.values())


# 16. hardlink fallback is safe (default copy_files=False should still produce
# readable files even if hardlinking isn't supported across the tmp filesystem)
def test_default_mode_produces_readable_files(tmp_path):
    source = _make_source_dataset(tmp_path)
    result = build_dataset(source, tmp_path / "out")

    any_output_file = next(result["output_dir"].rglob("*.jpg"))
    assert any_output_file.read_bytes() == b"fake-image-bytes"
