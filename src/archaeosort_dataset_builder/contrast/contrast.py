import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from archaeosort_dataset_builder.config.settings import settings


def contrast(dataset=None, low_threshold=30, high_threshold=80, max_side=1280):

    dataset = Path(dataset) if dataset else settings.dataset

    low = 0
    normal = 0
    high = 0
    unreadable = 0

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        try:
            with Image.open(img) as image:
                # Para JPEG solicita una decodificaciÃ³n reducida.
                image.draft("L", (max_side, max_side))
                image = image.convert("L")

                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side))

                gray = np.asarray(image, dtype=np.float32)

                std = float(gray.std())

                if std < low_threshold:
                    low += 1

                elif std > high_threshold:
                    high += 1

                else:
                    normal += 1

                del gray

        except (UnidentifiedImageError, OSError, MemoryError):
            unreadable += 1
            continue

    report = {
        "low": low,
        "normal": normal,
        "high": high,
        "unreadable": unreadable,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "max_analysis_side": max_side,
    }

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "contrast.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("CONTRAST ANALYSIS")
    print("=" * 60)
    print(f"Low        : {low}")
    print(f"Normal     : {normal}")
    print(f"High       : {high}")
    print(f"Unreadable : {unreadable}")


if __name__ == "__main__":
    contrast()

