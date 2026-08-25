"""Tests for preprocessing/object_crop.py using synthetic images (drawn with
OpenCV/numpy) so no real photos are needed. Real-photo validation was done
manually against sample stone_figshare images during development (see
project notes) — this suite locks in the logic: crop to the object, prefer
non-border contours over a border-touching scale bar/label, and fall back
gracefully when nothing confident is found.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from archaeosort_dataset_builder.preprocessing.object_crop import (
    CropError,
    crop_dataset,
    crop_object,
)


def _make_image(path, width=400, height=300, bg=230):
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    return img


def _save(img, path):
    cv2.imwrite(str(path), img)


def test_crop_object_isolates_a_centered_dark_blob(tmp_path):
    img = _make_image(tmp_path, width=400, height=300, bg=230)
    cv2.circle(img, (200, 150), 60, (40, 40, 40), thickness=-1)
    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    _save(img, src)

    result = crop_object(src, out)

    assert result["cropped"] is True
    assert result["low_confidence"] is False
    assert out.exists()
    # Cropped image should be much smaller than the original frame.
    cropped = cv2.imread(str(out))
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]


def test_crop_object_prefers_centered_object_over_border_ruler(tmp_path):
    img = _make_image(tmp_path, width=500, height=400, bg=235)
    # A "ruler" strip touching the left border.
    cv2.rectangle(img, (0, 0), (40, 400), (20, 20, 20), thickness=-1)
    # The actual specimen, centered, away from any border.
    cv2.circle(img, (300, 200), 70, (60, 60, 60), thickness=-1)

    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    _save(img, src)

    result = crop_object(src, out)

    assert result["cropped"] is True
    assert result["low_confidence"] is False
    x0, _y0, x1, _y1 = result["bbox"]
    # The crop should center on the specimen (~ x in [230, 370]), not the
    # ruler strip (x in [0, 40]).
    assert x0 > 100
    assert x1 > 300


def test_crop_object_falls_back_to_border_contour_when_nothing_else_found(tmp_path):
    img = _make_image(tmp_path, width=300, height=300, bg=235)
    # Only a border-touching shape exists — no centered object.
    cv2.rectangle(img, (0, 100), (300, 180), (30, 30, 30), thickness=-1)

    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    _save(img, src)

    result = crop_object(src, out)

    assert result["cropped"] is True
    assert result["low_confidence"] is True


def test_crop_object_keeps_original_when_no_contour_found(tmp_path):
    img = _make_image(tmp_path, width=200, height=200, bg=200)
    # Perfectly uniform image: nothing to threshold.
    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    _save(img, src)

    result = crop_object(src, out)

    assert result["cropped"] is False
    assert result["low_confidence"] is True
    assert out.exists()


def test_crop_object_raises_on_unreadable_image(tmp_path):
    bad = tmp_path / "not_an_image.jpg"
    bad.write_text("not a real jpeg")
    out = tmp_path / "out.jpg"

    with pytest.raises(CropError):
        crop_object(bad, out)


def _make_class_dataset(root):
    dataset = root / "dataset"
    for class_name, blob_color in (("class_a", (40, 40, 40)), ("class_b", (60, 60, 200))):
        class_dir = dataset / class_name
        class_dir.mkdir(parents=True)
        for i in range(3):
            img = _make_image(root, width=300, height=300, bg=230)
            cv2.circle(img, (150, 150), 50, blob_color, thickness=-1)
            _save(img, class_dir / f"img_{i}.jpg")
    return dataset


def test_crop_dataset_mirrors_class_layout_and_writes_report(tmp_path):
    source = _make_class_dataset(tmp_path)
    output = tmp_path / "cropped"

    report = crop_dataset(source, output)

    assert report["total"]["images"] == 6
    assert report["total"]["cropped"] == 6
    assert (output / "class_a").is_dir()
    assert (output / "class_b").is_dir()
    assert len(list((output / "class_a").glob("*.jpg"))) == 3
    assert (output / "crop_report.json").exists()


def test_crop_dataset_rejects_missing_source(tmp_path):
    with pytest.raises(CropError):
        crop_dataset(tmp_path / "does-not-exist", tmp_path / "out")


def test_crop_dataset_rejects_nonempty_output_without_force(tmp_path):
    source = _make_class_dataset(tmp_path)
    output = tmp_path / "cropped"
    output.mkdir()
    (output / "existing_file.txt").write_text("hi")

    with pytest.raises(CropError):
        crop_dataset(source, output)

    # force=True should proceed without raising.
    report = crop_dataset(source, output, force=True)
    assert report["total"]["images"] == 6
