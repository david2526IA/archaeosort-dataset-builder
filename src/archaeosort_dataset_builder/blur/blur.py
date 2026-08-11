import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.archaeosort_dataset_builder.config.settings import settings


def blur(dataset=None, threshold=100, max_side=1280):

    dataset = Path(dataset) if dataset else settings.dataset

    blurry = 0
    sharp = 0
    unreadable = 0

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        try:
            with Image.open(img) as image:
                image.draft("L", (max_side, max_side))
                image = image.convert("L")

                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side))

                gray = np.asarray(image, dtype=np.uint8)

                score = float(cv2.Laplacian(gray, cv2.CV_32F).var())

                if score < threshold:
                    blurry += 1
                else:
                    sharp += 1

                del gray

        except (UnidentifiedImageError, OSError, MemoryError):
            unreadable += 1
            continue

    report = {
        "sharp": sharp,
        "blurry": blurry,
        "unreadable": unreadable,
        "threshold": threshold,
        "max_analysis_side": max_side,
    }

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "blur.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("BLUR ANALYSIS")
    print("=" * 60)
    print(f"Sharp      : {sharp}")
    print(f"Blurry     : {blurry}")
    print(f"Unreadable : {unreadable}")


if __name__ == "__main__":
    blur()
