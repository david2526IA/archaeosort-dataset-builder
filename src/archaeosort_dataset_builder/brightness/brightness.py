import json
from pathlib import Path

import cv2
import numpy as np

from archaeosort_dataset_builder.config.settings import settings


def brightness(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    dark = 0
    normal = 0
    bright = 0

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        image = cv2.imread(str(img))

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        value = np.mean(gray)

        if value < 60:
            dark += 1
        elif value > 190:
            bright += 1
        else:
            normal += 1

    report = {"dark": dark, "normal": normal, "bright": bright}

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "brightness.json", "w") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("BRIGHTNESS ANALYSIS")
    print("=" * 60)
    print(f"Dark     : {dark}")
    print(f"Normal   : {normal}")
    print(f"Bright   : {bright}")

