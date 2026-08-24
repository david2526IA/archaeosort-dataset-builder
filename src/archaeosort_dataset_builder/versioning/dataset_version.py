"""Deterministic content fingerprint and manifest for a built dataset.

Fase E hardening, fixing three issues tracked in ARCHAEOSORT_AGENT_CONTEXT.md:

1. ``sha256_directory`` used to call ``file.read_bytes()``, loading each
   whole file into memory. It now reads in fixed-size chunks.
2. If ``manifest.json`` already existed in the directory (e.g. rebuilding
   the same output), it was included in its own hash, making the
   fingerprint depend on its own previous value. ``manifest.json`` (and any
   other file in ``EXCLUDED_FROM_HASH``) is now always excluded from the
   hash and from the file/size counts.
3. Deterministic contract: ``sha256`` depends ONLY on the relative paths
   and byte content of the dataset's files (sorted, so ordering is stable).
   It does NOT depend on wall-clock time. The ``version`` and ``created``
   fields remain timestamp-based on purpose (they record *when* the
   manifest was written), but they are metadata, not part of the
   fingerprint — two manifests written at different times for identical
   file content will always have the same ``sha256``.

Found during the real 5,319-image Fase F smoke test: the first version of
this fix only excluded ``manifest.json`` itself, so ``build``'s own
``split_report.json`` and ``provenance.json`` (which the *builder* writes
into the same output directory, not this module) were still being hashed
and counted as "dataset files" — inflating the file count by 2, and
breaking determinism outright, since ``provenance.json`` contains a
``created`` timestamp that differs on every build. Both are now excluded
too: the fingerprint and file/size counts describe the dataset content
(train/val/test), not the build's own bookkeeping files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Build bookkeeping files that must never influence the dataset's own
# content fingerprint or file/size counts. manifest.json is written by this
# module; split_report.json and provenance.json are written by
# builder.dataset_builder into the same output directory.
EXCLUDED_FROM_HASH = {"manifest.json", "split_report.json", "provenance.json"}


def _iter_hashable_files(directory: Path, exclude: set[str]):
    for file in sorted(directory.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(directory).as_posix()
        if relative in exclude:
            continue
        yield file, relative


def sha256_directory(directory: Path, exclude: set[str] | None = None) -> str:
    """Deterministic SHA-256 fingerprint of a directory's file contents.

    Reads each file in fixed-size chunks (never loads a whole file into
    memory) and hashes each file's relative path together with its bytes,
    in sorted path order, so the result only depends on file paths and
    content — never on modification time or on files listed in ``exclude``
    (defaults to ``EXCLUDED_FROM_HASH``, i.e. ``manifest.json`` itself).
    """
    exclude = EXCLUDED_FROM_HASH if exclude is None else exclude

    sha = hashlib.sha256()

    for file, relative in _iter_hashable_files(directory, exclude):
        sha.update(relative.encode("utf-8"))

        with file.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                sha.update(chunk)

    return sha.hexdigest()


def create_manifest(dataset_dir: Path) -> dict:
    total_files = 0
    total_size = 0

    for file, _relative in _iter_hashable_files(dataset_dir, EXCLUDED_FROM_HASH):
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


def save_manifest(dataset_dir: Path) -> Path:
    manifest = create_manifest(dataset_dir)

    output = dataset_dir / "manifest.json"

    with output.open("w", encoding="utf8") as f:
        json.dump(manifest, f, indent=4)

    print("=" * 60)
    print("ARCHAEOSORT DATASET VERSION")
    print("=" * 60)
    print(json.dumps(manifest, indent=4))
    print("=" * 60)

    return output


if __name__ == "__main__":
    save_manifest(Path("exports/imagefolder"))
