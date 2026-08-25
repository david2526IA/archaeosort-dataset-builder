"""Object-isolation preprocessing: crop archaeology specimen photos down to
just the object, removing the studio background and any scale-bar/label card.

Why this exists (2026-08-24): a ResNet18 baseline trained on the raw
``stone_figshare`` photos hit 100% test accuracy after only 2 epochs.
Inspecting sample images showed the two classes were photographed with
different, internally-consistent studio setups (``natural_rock``: uniform
gray background + black/white cm scale bar; ``stone_artefact``: several
different backgrounds and different scale-bar/label card styles). A CNN can
trivially separate the classes by background/scale-bar style alone, without
learning anything about the object itself — a classic shortcut-learning /
dataset-bias failure mode, not a bug in the dataset engineering pipeline.

This module removes that confound with classical computer vision (no ML, no
training data needed, nothing to overfit): threshold the (mostly uniform)
background, find the object's contour, and crop to its bounding box plus a
small margin. Scale bars and label cards are usually placed at or touching
the frame's edge, while the specimen itself is normally floated away from
the border — so border-touching contours are only used as a last resort.

This is a heuristic, not a perfect segmentation: it assumes a single
dominant object on a roughly uniform background, which held for every
sample inspected in ``stone_figshare`` but may not hold for every future
dataset. Always check ``crop_report.json`` (``low_confidence`` count) after
running ``crop_dataset`` on a new dataset before trusting it blindly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BORDER_TOLERANCE_PX = 3
MIN_AREA_FRAC = 0.01
MAX_AREA_FRAC = 0.6
DEFAULT_MARGIN_FRAC = 0.06


class CropError(ValueError):
    """Raised when the source/output layout for crop_dataset is invalid."""


def _touches_border(contour, width: int, height: int, tolerance: int = BORDER_TOLERANCE_PX) -> bool:
    bx, by, bw, bh = cv2.boundingRect(contour)
    return bx <= tolerance or by <= tolerance or bx + bw >= width - tolerance or by + bh >= height - tolerance


def _find_object_contour(gray: np.ndarray):
    """Return (contour, area_fraction, used_border_fallback) or None."""
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh_dark_fg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_light_fg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    fallback = None

    for thresh in (thresh_dark_fg, thresh_light_fg):
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            area_frac = area / (height * width)

            if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
                continue

            if _touches_border(contour, width, height):
                if fallback is None or area > fallback[1]:
                    fallback = (contour, area, area_frac)
                continue

            if best is None or area > best[1]:
                best = (contour, area, area_frac)
            break  # only the largest valid contour per threshold polarity

    if best is not None:
        return best[0], best[2], False
    if fallback is not None:
        return fallback[0], fallback[2], True
    return None


def crop_object(image_path: Path, output_path: Path, margin_frac: float = DEFAULT_MARGIN_FRAC) -> dict[str, Any]:
    """Crop a single specimen photo to just the object. Writes the result
    (cropped, or the original untouched if no confident contour was found)
    to ``output_path`` and returns a dict describing what happened."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise CropError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found = _find_object_contour(gray)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if found is None:
        cv2.imwrite(str(output_path), image)
        return {
            "cropped": False,
            "low_confidence": True,
            "reason": "no confident object contour found; kept original",
            "area_frac": None,
            "used_border_fallback": None,
        }

    contour, area_frac, used_border_fallback = found
    x, y, box_w, box_h = cv2.boundingRect(contour)

    margin_x = int(box_w * margin_frac)
    margin_y = int(box_h * margin_frac)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(width, x + box_w + margin_x)
    y1 = min(height, y + box_h + margin_y)

    cropped = image[y0:y1, x0:x1]
    cv2.imwrite(str(output_path), cropped)

    return {
        "cropped": True,
        "low_confidence": used_border_fallback,
        "reason": "used a border-touching contour as a fallback" if used_border_fallback else None,
        "area_frac": round(area_frac, 4),
        "used_border_fallback": used_border_fallback,
        "bbox": [x0, y0, x1, y1],
        "original_size": [width, height],
        "cropped_size": [x1 - x0, y1 - y0],
    }


def crop_dataset(
    source: Path,
    output: Path,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
    force: bool = False,
) -> dict[str, Any]:
    """Crop every image in a class-folder dataset (``source/<class>/*.jpg``)
    to just the object, mirroring the same ``<class>/`` layout under
    ``output``. Writes ``output/crop_report.json`` summarizing results (per
    class and overall) and returns the same summary dict.

    Raises CropError if ``source`` doesn't exist, has no class subfolders,
    or if ``output`` already exists and is non-empty without ``force=True``.
    """
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    if not source.is_dir():
        raise CropError(f"Source dataset not found: {source}")

    class_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if not class_dirs:
        raise CropError(f"No class subfolders found under {source}")

    if output.exists() and any(output.iterdir()) and not force:
        raise CropError(f"Output directory is not empty: {output} (use force=True to overwrite)")

    output.mkdir(parents=True, exist_ok=True)

    per_class: dict[str, Any] = {}
    total = {"images": 0, "cropped": 0, "low_confidence": 0, "unreadable": 0}

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_output = output / class_name
        stats = {"images": 0, "cropped": 0, "low_confidence": 0, "unreadable": 0}

        image_paths = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            stats["images"] += 1
            out_path = class_output / image_path.name
            try:
                result = crop_object(image_path, out_path, margin_frac=margin_frac)
            except CropError:
                stats["unreadable"] += 1
                continue

            if result["cropped"]:
                stats["cropped"] += 1
            if result["low_confidence"]:
                stats["low_confidence"] += 1

        per_class[class_name] = stats
        for key in total:
            total[key] += stats[key]

    report = {
        "source": str(source),
        "output": str(output),
        "margin_frac": margin_frac,
        "classes": per_class,
        "total": total,
    }

    report_path = output / "crop_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = report_path

    return report
