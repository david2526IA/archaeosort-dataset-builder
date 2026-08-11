from pathlib import Path

from archaeosort_dataset_builder.validator.class_validator import validate_classes
from archaeosort_dataset_builder.validator.image_validator import validate_image
from archaeosort_dataset_builder.validator.metadata_validator import validate_metadata

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def validate_dataset(dataset_root: Path) -> list[str]:
    issues: list[str] = []

    issues.extend(validate_metadata(dataset_root))
    issues.extend(validate_classes(dataset_root))

    for image_path in dataset_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            issues.extend(validate_image(image_path))

    return issues
