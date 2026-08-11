from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
]


def validate_metadata(dataset_root: Path) -> list[str]:
    issues: list[str] = []

    for filename in REQUIRED_FILES:
        if not (dataset_root / filename).exists():
            issues.append(f"Missing metadata file: {filename}")

    return issues
