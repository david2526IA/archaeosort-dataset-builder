import json
from pathlib import Path

from PIL import Image

from src.archaeosort_dataset_builder.config.settings import settings


def aspect_ratio(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    square = 0
    landscape = 0
    portrait = 0

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        try:
            with Image.open(img) as im:
                w, h = im.size

                ratio = w / h

                if 0.9 <= ratio <= 1.1:
                    square += 1

                elif ratio > 1.1:
                    landscape += 1

                else:
                    portrait += 1

        except (OSError, ValueError):
            continue

    report = {"square": square, "landscape": landscape, "portrait": portrait}

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "aspect_ratio.json", "w") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("ASPECT RATIO")
    print("=" * 60)
    print(f"Square      : {square}")
    print(f"Landscape   : {landscape}")
    print(f"Portrait    : {portrait}")
