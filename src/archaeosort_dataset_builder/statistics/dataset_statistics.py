from pathlib import Path
from collections import Counter


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def compute_statistics(dataset_root: Path) -> dict:

    total_images = 0

    class_counter = Counter()

    extension_counter = Counter()

    for file in dataset_root.rglob("*"):

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1

        class_counter[file.parent.name] += 1

        extension_counter[file.suffix.lower()] += 1

    return {
        "total_images": total_images,
        "classes": dict(class_counter),
        "extensions": dict(extension_counter),
    }
