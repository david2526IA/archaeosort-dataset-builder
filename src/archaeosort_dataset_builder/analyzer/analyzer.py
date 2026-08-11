import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from src.archaeosort_dataset_builder.config.settings import settings


def analyze(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    widths = []
    heights = []

    formats = Counter()

    total = 0

    for img_path in dataset.rglob("*"):
        if img_path.suffix.lower() not in settings.image_extensions:
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size

                widths.append(w)
                heights.append(h)

                formats[img.format] += 1

                total += 1

        except (OSError, ValueError):
            continue

    print("=" * 60)
    print("DATASET ANALYZER")
    print("=" * 60)

    print(f"Images           : {total}")
    print(f"Average width    : {statistics.mean(widths):.1f}")
    print(f"Average height   : {statistics.mean(heights):.1f}")

    print(f"Min width        : {min(widths)}")
    print(f"Max width        : {max(widths)}")

    print(f"Min height       : {min(heights)}")
    print(f"Max height       : {max(heights)}")

    print()

    print("Formats")

    for k, v in formats.items():
        print(f"{k:10} {v}")
