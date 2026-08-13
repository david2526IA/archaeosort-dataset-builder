from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from archaeosort_dataset_builder.config.settings import settings


def scan_quality(
    dataset: str | Path | None = None,
    blur_threshold: float = 100,
    dark_threshold: float = 60,
    bright_threshold: float = 190,
    low_contrast_threshold: float = 30,
    high_contrast_threshold: float = 80,
    max_side: int = 1280,
) -> dict:
    """Analyze blur, brightness and contrast in a single image pass."""

    dataset_path = Path(dataset) if dataset else settings.require_dataset()

    blur_counts = {
        "sharp": 0,
        "blurry": 0,
        "unreadable": 0,
    }

    brightness_counts = {
        "dark": 0,
        "normal": 0,
        "bright": 0,
    }

    contrast_counts = {
        "low": 0,
        "normal": 0,
        "high": 0,
        "unreadable": 0,
    }

    analyzed = 0

    for image_path in dataset_path.rglob("*"):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in settings.image_extensions
        ):
            continue

        try:
            with Image.open(image_path) as image:
                image.draft("L", (max_side, max_side))
                image = image.convert("L")

                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side))

                gray = np.asarray(image, dtype=np.uint8)

            analyzed += 1

            brightness_value = float(gray.mean())
            contrast_value = float(gray.std())
            blur_value = float(cv2.Laplacian(gray, cv2.CV_32F).var())

            if blur_value < blur_threshold:
                blur_counts["blurry"] += 1
            else:
                blur_counts["sharp"] += 1

            if brightness_value < dark_threshold:
                brightness_counts["dark"] += 1
            elif brightness_value > bright_threshold:
                brightness_counts["bright"] += 1
            else:
                brightness_counts["normal"] += 1

            if contrast_value < low_contrast_threshold:
                contrast_counts["low"] += 1
            elif contrast_value > high_contrast_threshold:
                contrast_counts["high"] += 1
            else:
                contrast_counts["normal"] += 1

        except (UnidentifiedImageError, OSError, MemoryError, ValueError):
            blur_counts["unreadable"] += 1
            contrast_counts["unreadable"] += 1

    blur_report = {
        **blur_counts,
        "threshold": blur_threshold,
        "max_analysis_side": max_side,
    }

    brightness_report = {
        **brightness_counts,
        "dark_threshold": dark_threshold,
        "bright_threshold": bright_threshold,
        "max_analysis_side": max_side,
    }

    contrast_report = {
        **contrast_counts,
        "low_threshold": low_contrast_threshold,
        "high_threshold": high_contrast_threshold,
        "max_analysis_side": max_side,
    }

    settings.reports.mkdir(parents=True, exist_ok=True)

    (settings.reports / "blur.json").write_text(
        json.dumps(blur_report, indent=4),
        encoding="utf8",
    )

    (settings.reports / "brightness.json").write_text(
        json.dumps(brightness_report, indent=4),
        encoding="utf8",
    )

    (settings.reports / "contrast.json").write_text(
        json.dumps(contrast_report, indent=4),
        encoding="utf8",
    )

    print("=" * 60)
    print("UNIFIED QUALITY SCAN")
    print("=" * 60)
    print(f"Images analyzed : {analyzed}")
    print()
    print(f"Sharp            : {blur_counts['sharp']}")
    print(f"Blurry           : {blur_counts['blurry']}")
    print()
    print(f"Dark             : {brightness_counts['dark']}")
    print(f"Normal brightness: {brightness_counts['normal']}")
    print(f"Bright           : {brightness_counts['bright']}")
    print()
    print(f"Low contrast     : {contrast_counts['low']}")
    print(f"Normal contrast  : {contrast_counts['normal']}")
    print(f"High contrast    : {contrast_counts['high']}")

    return {
        "images": analyzed,
        "blur": blur_report,
        "brightness": brightness_report,
        "contrast": contrast_report,
    }
