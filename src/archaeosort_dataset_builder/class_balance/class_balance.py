from __future__ import annotations

import json
from pathlib import Path

from archaeosort_dataset_builder.config.settings import settings

SPLIT_NAMES = {"train", "val", "test"}


def _count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in settings.image_extensions
    )


def compute_class_balance(dataset: Path) -> dict:
    dataset = Path(dataset)

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    directories = [path for path in dataset.iterdir() if path.is_dir()]
    directory_names = {path.name.lower() for path in directories}

    classes: dict[str, int] = {}

    # Layout A:
    # dataset/train/class_a
    # dataset/val/class_a
    # dataset/test/class_a
    if directory_names & SPLIT_NAMES:
        layout = "split"

        for split_name in ("train", "val", "test"):
            split_dir = dataset / split_name

            if not split_dir.is_dir():
                continue

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                count = _count_images(class_dir)
                classes[class_dir.name] = classes.get(class_dir.name, 0) + count

    # Layout B:
    # dataset/class_a
    # dataset/class_b
    else:
        layout = "class_folders"

        for class_dir in directories:
            count = _count_images(class_dir)

            if count > 0:
                classes[class_dir.name] = count

    total = sum(classes.values())

    return {
        "layout": layout,
        "classes": classes,
        "total": total,
    }


def class_balance(dataset=None):
    dataset_path = Path(dataset) if dataset else settings.dataset

    report = compute_class_balance(dataset_path)

    settings.reports.mkdir(parents=True, exist_ok=True)

    output = settings.reports / "class_balance.json"
    output.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf8",
    )

    print("=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)
    print(f"Layout : {report['layout']}")
    print(f"Total  : {report['total']}")
    print()

    total = report["total"]

    for cls, count in sorted(report["classes"].items()):
        pct = (count / total * 100) if total else 0
        print(f"{cls:25} {count:5d} ({pct:.2f}%)")

    return report
