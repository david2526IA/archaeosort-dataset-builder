from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_directory(directory: Path) -> str:
    sha = hashlib.sha256()

    for file in sorted(directory.rglob("*")):
        if file.is_file():
            sha.update(file.relative_to(directory).as_posix().encode())
            sha.update(file.read_bytes())

    return sha.hexdigest()


def create_manifest(dataset_dir: Path) -> dict:

    total_files = 0
    total_size = 0

    for file in dataset_dir.rglob("*"):
        if file.is_file():
            total_files += 1
            total_size += file.stat().st_size

    return {
        "version": datetime.now(UTC).strftime("%Y.%m.%d.%H%M%S"),
        "created": datetime.now(UTC).isoformat(),
        "dataset": dataset_dir.name,
        "files": total_files,
        "size_bytes": total_size,
        "sha256": sha256_directory(dataset_dir),
    }


def save_manifest(dataset_dir: Path):

    manifest = create_manifest(dataset_dir)

    output = dataset_dir / "manifest.json"

    with output.open("w", encoding="utf8") as f:
        json.dump(manifest, f, indent=4)

    print("=" * 60)
    print("ARCHAEOSORT DATASET VERSION")
    print("=" * 60)
    print(json.dumps(manifest, indent=4))
    print("=" * 60)


if __name__ == "__main__":
    save_manifest(Path("exports/imagefolder"))
