"""One-shot builder for the "robust" 6-class dataset (natural_rock,
stone_artefact, natural_bone, bone_artefact, natural_wood, wood_artefact),
used to train a real multi-class ResNet18 classifier (classifier/baseline.py)
instead of the earlier zero-shot/prototype few-shot approaches.

Steps:
  1. Download a much larger reference set for the 4 bone/wood categories
     via reference_dataset.build_reference_dataset (Met Museum + Wikimedia
     Commons + iNaturalist -- see that module's docstring).
  2. Pool natural_rock/stone_artefact from the existing, already-large
     Auckland figshare dataset (datasets/stone_figshare_cropped/) -- no
     need to re-download or re-scrape those, there's already ~5,300
     labelled images sitting on disk.
  3. Re-split everything together into one train/val/test dataset at
     datasets/robust_v1/, using the project's existing splitter so the
     ratios/seed/hardlinking behaviour matches the rest of the codebase.

Run from the project root with the real environment active:
    uv run --no-sync python scripts/build_robust_dataset.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from archaeosort_dataset_builder.prototypes.reference_dataset import (
    DEFAULT_SOURCES,
    build_reference_dataset,
)
from archaeosort_dataset_builder.splitter.dataset_splitter import (
    build_split_dataset,
    discover_images_by_class,
    prepare_output_directory,
    save_report,
)

# INFO-level heartbeat from reference_dataset.py's fetch loops -- makes a
# long download show live, visible progress in the redirected log instead
# of going silent until it finishes (or crashes without a trace).
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")

# Overridable via env var so an "enrichment" re-run can raise the target
# without editing this file -- build_reference_dataset's top-up logic
# (existing images below target get extra, not re-downloaded) means this
# is safe to raise and re-run at any time.
PER_CLASS_TARGET = int(os.environ.get("PER_CLASS_TARGET", "250"))
REFERENCE_DIR = Path("datasets/prototype_references_large")
STONE_DATASET_DIR = Path("datasets/stone_figshare_cropped/v1")
RAW_POOL_DIR = Path("datasets/robust_v1_raw")
OUTPUT_DIR = Path("datasets/robust_v1")
STONE_CLASSES = ("natural_rock", "stone_artefact")


def pool_stone_images() -> None:
    """Hardlink every natural_rock/stone_artefact image (across the
    existing train/val/test splits) into one flat pool per class, so the
    splitter below re-shuffles everything together with the bone/wood
    classes rather than reusing the old stone-only split boundaries."""
    for class_name in STONE_CLASSES:
        dest_dir = RAW_POOL_DIR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for split in ("train", "val", "test"):
            source_dir = STONE_DATASET_DIR / split / class_name
            if not source_dir.is_dir():
                continue
            for image_path in source_dir.iterdir():
                if not image_path.is_file():
                    continue
                dest_path = dest_dir / f"{split}_{image_path.name}"
                try:
                    dest_path.hardlink_to(image_path.resolve())
                except (OSError, FileExistsError):
                    dest_path.write_bytes(image_path.read_bytes())
                count += 1
        print(f"[stone] pooled {count} images for {class_name}", flush=True)


def pool_reference_images() -> None:
    """Copy the downloaded bone/wood reference images into the same raw
    pool layout the stone classes use."""
    for class_name in DEFAULT_SOURCES:
        source_dir = REFERENCE_DIR / class_name
        dest_dir = RAW_POOL_DIR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        if source_dir.is_dir():
            for image_path in source_dir.iterdir():
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                dest_path = dest_dir / image_path.name
                try:
                    dest_path.hardlink_to(image_path.resolve())
                except (OSError, FileExistsError):
                    dest_path.write_bytes(image_path.read_bytes())
                count += 1
        print(f"[reference] pooled {count} images for {class_name}", flush=True)


def main() -> None:
    print(f"Step 1/3: downloading reference images (target {PER_CLASS_TARGET}/class)...", flush=True)
    # force=False: a category that already has images (from a previous,
    # interrupted run) is reused as-is instead of re-downloaded -- this
    # run is resumable rather than starting over from zero every retry.
    manifest = build_reference_dataset(
        REFERENCE_DIR,
        sources=DEFAULT_SOURCES,
        per_class=PER_CLASS_TARGET,
        force=False,
    )
    print(json.dumps({k: v["count"] for k, v in manifest["categories"].items()}, indent=2), flush=True)

    print("Step 2/3: pooling stone + bone/wood images into one raw layout...", flush=True)
    pool_stone_images()
    pool_reference_images()

    print("Step 3/3: splitting the pooled 6-class dataset into train/val/test...", flush=True)
    images_by_class = discover_images_by_class(RAW_POOL_DIR)
    for class_name, images in sorted(images_by_class.items()):
        print(f"  {class_name}: {len(images)} images total", flush=True)

    # Always rebuild OUTPUT_DIR from scratch: re-running over a stale
    # previous split led transfer_file to hit "destination already
    # exists" -> hardlink fails -> falls back to shutil.copy2 -> can hit a
    # transient PermissionError if OneDrive/AV briefly locks the file that
    # was just (re)written a moment earlier, aborting the whole build
    # partway through with an inconsistent, half-updated dataset on disk.
    # A clean wipe means every transfer_file call hardlinks into a
    # brand-new path, never colliding with anything.
    prepare_output_directory(OUTPUT_DIR, force=True)

    report = build_split_dataset(
        images_by_class=images_by_class,
        output_dir=OUTPUT_DIR,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        copy_files=False,
    )
    report_path = save_report(report, OUTPUT_DIR)

    print(f"Done. Dataset at: {OUTPUT_DIR.resolve()}")
    print(f"Split report: {report_path.resolve()}")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
