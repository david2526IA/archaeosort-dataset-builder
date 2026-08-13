from collections import defaultdict
from pathlib import Path

from archaeosort_dataset_builder.config.settings import settings


def size_index(dataset=None):

    dataset = Path(dataset) if dataset else settings.dataset

    index = defaultdict(list)

    for img in dataset.rglob("*"):
        if img.suffix.lower() not in settings.image_extensions:
            continue

        index[img.stat().st_size].append(img)

    return index

