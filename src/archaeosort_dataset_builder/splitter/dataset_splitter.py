from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide un dataset de imágenes en train, val y test."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Los ratios deben sumar 1.0. "
            f"Resultado actual: {total:.6f}"
        )

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Los ratios no pueden ser negativos.")


def discover_images_by_class(
    input_dir: Path,
) -> dict[str, list[Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {input_dir.resolve()}"
        )

    images_by_class: dict[str, list[Path]] = defaultdict(list)

    for image_path in input_dir.rglob("*"):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            class_name = image_path.parent.name
            images_by_class[class_name].append(image_path)

    if not images_by_class:
        raise ValueError("No se encontraron imágenes clasificadas.")

    return dict(images_by_class)


def split_class_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def prepare_output_directory(
    output_dir: Path,
    force: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                "La carpeta de salida ya contiene archivos. "
                "Usa --force para regenerarla."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def transfer_file(
    source_path: Path,
    destination_path: Path,
    copy_files: bool,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if copy_files:
        shutil.copy2(source_path, destination_path)
    else:
        try:
            destination_path.hardlink_to(source_path.resolve())
        except OSError:
            shutil.copy2(source_path, destination_path)


def build_split_dataset(
    images_by_class: dict[str, list[Path]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    copy_files: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "seed": seed,
        "mode": "copy" if copy_files else "hardlink_or_copy",
        "splits": {
            "train": {},
            "val": {},
            "test": {},
        },
        "total_images": 0,
    }

    for class_index, class_name in enumerate(sorted(images_by_class)):
        split_map = split_class_images(
            images=images_by_class[class_name],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + class_index,
        )

        for split_name, image_paths in split_map.items():
            report["splits"][split_name][class_name] = len(image_paths)
            report["total_images"] += len(image_paths)

            for index, image_path in enumerate(image_paths):
                destination_name = (
                    f"{index:06d}_{image_path.name}"
                )
                destination_path = (
                    output_dir
                    / split_name
                    / class_name
                    / destination_name
                )

                transfer_file(
                    source_path=image_path,
                    destination_path=destination_path,
                    copy_files=copy_files,
                )

    return report


def save_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "split_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return report_path


def main() -> None:
    args = parse_args()

    validate_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    prepare_output_directory(
        args.output_dir,
        args.force,
    )

    images_by_class = discover_images_by_class(args.input_dir)

    report = build_split_dataset(
        images_by_class=images_by_class,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_files=args.copy,
    )

    report_path = save_report(report, args.output_dir)

    print()
    print("=" * 60)
    print("ARCHAEOSORT DATASET SPLITTER")
    print("=" * 60)
    print(f"Total images: {report['total_images']}")
    print(f"Classes: {len(images_by_class)}")

    for split_name, class_counts in report["splits"].items():
        split_total = sum(class_counts.values())
        print(f"{split_name}: {split_total}")

    print(f"Output: {args.output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
