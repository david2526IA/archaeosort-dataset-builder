import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.archaeosort_dataset_builder.config.settings import settings


def verify(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    extensions = Counter()

    corrupted = 0
    empty = 0
    images = 0
    directories = 0

    for path in dataset.rglob("*"):
        if path.is_dir():
            directories += 1
            continue

        if path.stat().st_size == 0:
            empty += 1
            continue

        if path.suffix.lower() not in settings.image_extensions:
            continue

        images += 1
        extensions[path.suffix.lower()] += 1

        try:
            with Image.open(path) as img:
                img.verify()

        except (OSError, ValueError):
            corrupted += 1

    report = {
        "dataset": str(dataset),
        "exists": dataset.exists(),
        "directories": directories,
        "images": images,
        "empty_files": empty,
        "corrupted_images": corrupted,
        "extensions": dict(extensions),
    }

    settings.reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = settings.reports / "verify_report.json"

    with report_path.open("w", encoding="utf8") as f:
        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("VERIFY REPORT")
    print("=" * 60)
    print(f"Dataset            : {dataset}")
    print(f"Directories        : {directories}")
    print(f"Images             : {images}")
    print(f"Empty files        : {empty}")
    print(f"Corrupted images   : {corrupted}")
    print()

    for ext, count in sorted(extensions.items()):
        print(f"{ext:8} {count}")

    print()
    print(f"Report saved to: {report_path}")

    return report
