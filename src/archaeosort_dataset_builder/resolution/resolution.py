import json
from pathlib import Path

from PIL import Image

from archaeosort_dataset_builder.config.settings import settings


def resolution(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    tiny = 0
    small = 0
    medium = 0
    large = 0

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        try:
            with Image.open(img) as im:
                w, h = im.size

                area = w * h

                if area < 256 * 256:
                    tiny += 1
                elif area < 512 * 512:
                    small += 1
                elif area < 1024 * 1024:
                    medium += 1
                else:
                    large += 1

        except (OSError, ValueError):
            continue

    report = {"tiny": tiny, "small": small, "medium": medium, "large": large}

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "resolution.json", "w") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("RESOLUTION ANALYSIS")
    print("=" * 60)
    print(f"Tiny    : {tiny}")
    print(f"Small   : {small}")
    print(f"Medium  : {medium}")
    print(f"Large   : {large}")

