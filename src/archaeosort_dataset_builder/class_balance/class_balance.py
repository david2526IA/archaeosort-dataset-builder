import json
from pathlib import Path

from src.archaeosort_dataset_builder.config.settings import settings


def class_balance(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    classes = {}

    for split in ["train", "val", "test"]:
        split_dir = dataset / split

        if not split_dir.exists():
            continue

        for cls in split_dir.iterdir():
            if not cls.is_dir():
                continue

            count = 0

            for img in cls.rglob("*"):
                if img.suffix.lower() in settings.image_extensions:
                    count += 1

            classes[cls.name] = classes.get(cls.name, 0) + count

    total = sum(classes.values())

    report = {"classes": classes, "total": total}

    settings.reports.mkdir(parents=True, exist_ok=True)

    with open(settings.reports / "class_balance.json", "w") as f:
        json.dump(report, f, indent=4)

    print("=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)

    for cls, count in sorted(classes.items()):
        pct = (count / total * 100) if total else 0

        print(f"{cls:25} {count:5d} ({pct:.2f}%)")
