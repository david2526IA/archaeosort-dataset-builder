from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga fuentes Kaggle definidas en el manifiesto."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sources.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/downloads"),
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/metadata"),
    )
    return parser.parse_args()


def load_manifest(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download_kaggle_source(
    source: dict[str, Any],
    output_dir: Path,
) -> Path:
    source_dir = output_dir / source["id"]
    source_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        source["reference"],
        "-p",
        str(source_dir),
        "--force",
    ]

    print(f"[INFO] Descargando: {source['name']}")
    subprocess.run(command, check=True)

    zip_files = list(source_dir.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(
            f"No se encontró ningún ZIP en {source_dir.resolve()}"
        )

    return zip_files[0]


def save_metadata(
    source: dict[str, Any],
    downloaded_file: Path,
    metadata_dir: Path,
) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_id": source["id"],
        "source_name": source["name"],
        "source_type": source["source_type"],
        "reference": source["reference"],
        "license": source["license"],
        "downloaded_file": str(downloaded_file.resolve()),
        "file_size_bytes": downloaded_file.stat().st_size,
        "sha256": calculate_sha256(downloaded_file),
    }

    metadata_path = metadata_dir / f"{source['id']}.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return metadata_path


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {args.config.resolve()}"
        )

    manifest = load_manifest(args.config)

    kaggle_sources = [
        source
        for source in manifest["sources"]
        if source.get("enabled", False)
        and source.get("automatic_download", False)
        and source.get("source_type") == "kaggle"
    ]

    if not kaggle_sources:
        print("[INFO] No hay fuentes Kaggle activas.")
        return

    for source in kaggle_sources:
        downloaded_file = download_kaggle_source(
            source=source,
            output_dir=args.output_dir,
        )

        metadata_path = save_metadata(
            source=source,
            downloaded_file=downloaded_file,
            metadata_dir=args.metadata_dir,
        )

        print(f"[OK] Archivo: {downloaded_file.resolve()}")
        print(f"[OK] Metadatos: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
