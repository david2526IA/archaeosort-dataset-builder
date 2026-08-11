from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae las fuentes descargadas del Dataset Builder."
    )

    parser.add_argument(
        "--downloads-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def discover_zip_files(downloads_dir: Path) -> list[Path]:
    if not downloads_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de descargas: {downloads_dir.resolve()}")

    return sorted(downloads_dir.glob("*/*.zip"))


def count_contents(destination: Path) -> tuple[int, int]:
    file_count = sum(1 for path in destination.rglob("*") if path.is_file())

    directory_count = sum(1 for path in destination.rglob("*") if path.is_dir())

    return file_count, directory_count


def extract_zip(
    source_id: str,
    zip_path: Path,
    staging_dir: Path,
    force: bool,
) -> dict[str, Any]:
    destination = staging_dir / source_id
    marker_path = destination / ".extraction_complete"

    if marker_path.exists() and not force:
        file_count, directory_count = count_contents(destination)

        return {
            "source_id": source_id,
            "status": "skipped",
            "reason": "already_extracted",
            "zip_file": str(zip_path.resolve()),
            "destination": str(destination.resolve()),
            "file_count": file_count,
            "directory_count": directory_count,
        }

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupt_member = archive.testzip()

        if corrupt_member is not None:
            raise zipfile.BadZipFile(f"Archivo corrupto dentro del ZIP: {corrupt_member}")

        archive.extractall(destination)

    marker_path.write_text(
        datetime.now(UTC).isoformat(),
        encoding="utf-8",
    )

    file_count, directory_count = count_contents(destination)

    return {
        "source_id": source_id,
        "status": "extracted",
        "zip_file": str(zip_path.resolve()),
        "zip_size_bytes": zip_path.stat().st_size,
        "destination": str(destination.resolve()),
        "file_count": file_count,
        "directory_count": directory_count,
        "top_level_entries": sorted(
            path.name for path in destination.iterdir() if path.name != ".extraction_complete"
        ),
        "extracted_at_utc": datetime.now(UTC).isoformat(),
    }


def save_report(
    source_id: str,
    report: dict[str, Any],
    metadata_dir: Path,
) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)

    report_path = metadata_dir / f"{source_id}_extraction.json"

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

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)

    zip_files = discover_zip_files(args.downloads_dir)

    if not zip_files:
        print("[INFO] No se encontraron archivos ZIP.")
        return

    print()
    print("=" * 60)
    print("ARCHAEOSORT DATASET BUILDER - EXTRACT SOURCES")
    print("=" * 60)

    for zip_path in zip_files:
        source_id = zip_path.parent.name

        print(f"[INFO] Fuente: {source_id}")
        print(f"[INFO] ZIP: {zip_path.resolve()}")

        report = extract_zip(
            source_id=source_id,
            zip_path=zip_path,
            staging_dir=args.staging_dir,
            force=args.force,
        )

        report_path = save_report(
            source_id=source_id,
            report=report,
            metadata_dir=args.metadata_dir,
        )

        print(f"[OK] Estado: {report['status']}")
        print(f"[OK] Archivos: {report['file_count']}")
        print(f"[OK] Carpetas: {report['directory_count']}")
        print(f"[OK] Destino: {report['destination']}")
        print(f"[OK] Informe: {report_path.resolve()}")
        print("-" * 60)


if __name__ == "__main__":
    main()
