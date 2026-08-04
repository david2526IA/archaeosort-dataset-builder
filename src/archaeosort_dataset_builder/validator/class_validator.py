from pathlib import Path


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def validate_classes(dataset_root: Path) -> list[str]:
    issues: list[str] = []

    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    if not class_dirs:
        issues.append("Dataset has no class folders.")
        return issues

    for class_dir in class_dirs:

        image_count = sum(
            1
            for file in class_dir.rglob("*")
            if file.suffix.lower() in VALID_EXTENSIONS
        )

        if image_count == 0:
            issues.append(f"Class '{class_dir.name}' is empty.")

    return issues
