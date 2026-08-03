from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SOURCE_FIELDS = {
    "id",
    "name",
    "source_type",
    "status",
    "enabled",
    "automatic_download",
    "license",
    "license_review_required",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida el manifiesto de fuentes de ArchaeoSort Dataset Builder."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sources.yaml"),
        help="Ruta al manifiesto YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manifest_validation.json"),
        help="Ruta del informe JSON.",
    )
    return parser.parse_args()


def load_manifest(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8-sig") as file:
        manifest = yaml.safe_load(file)

    if not isinstance(manifest, dict):
        raise ValueError("El manifiesto debe contener un objeto YAML principal.")

    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for section in (
        "project",
        "target_classes",
        "sources",
        "pipeline",
        "splits",
    ):
        if section not in manifest:
            errors.append(f"Falta la sección obligatoria: {section}")

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    project = manifest["project"]
    target_classes = manifest["target_classes"]
    sources = manifest["sources"]
    splits = manifest["splits"]

    declared_total = int(project.get("target_total_images", 0))

    calculated_total = sum(
        int(class_config.get("target_images", 0))
        for class_config in target_classes.values()
    )

    if declared_total != calculated_total:
        errors.append(
            "El objetivo total declarado no coincide con la suma de clases: "
            f"{declared_total} != {calculated_total}"
        )

    if not target_classes:
        errors.append("No se han definido clases objetivo.")

    if not isinstance(sources, list) or not sources:
        errors.append("La lista de fuentes está vacía o no es válida.")
        sources = []

    source_ids: set[str] = set()

    for index, source in enumerate(sources, start=1):
        source_name = source.get("id", f"source_{index}")

        missing_fields = REQUIRED_SOURCE_FIELDS - set(source)

        if missing_fields:
            errors.append(
                f"La fuente {source_name} no contiene: "
                f"{', '.join(sorted(missing_fields))}"
            )

        source_id = source.get("id")

        if source_id in source_ids:
            errors.append(f"ID de fuente duplicado: {source_id}")
        elif source_id:
            source_ids.add(source_id)

        automatic_download = bool(
            source.get("automatic_download", False)
        )
        review_required = bool(
            source.get("license_review_required", True)
        )

        if automatic_download and review_required:
            errors.append(
                f"La fuente {source_name} permite descarga automática "
                "pero su licencia requiere revisión."
            )

        if source.get("enabled") and source.get("status") != "available":
            warnings.append(
                f"La fuente {source_name} está activa pero su estado es "
                f"{source.get('status')}."
            )

        if source.get("license") in {
            None,
            "",
            "unknown",
            "pending_review",
        }:
            warnings.append(
                f"La fuente {source_name} no tiene una licencia confirmada."
            )

    train_ratio = float(splits.get("train", 0))
    validation_ratio = float(splits.get("validation", 0))
    test_ratio = float(splits.get("test", 0))
    split_total = train_ratio + validation_ratio + test_ratio

    if abs(split_total - 1.0) > 1e-9:
        errors.append(
            "Las particiones no suman 1.0: "
            f"{split_total:.6f}"
        )

    if not splits.get("prevent_source_leakage", False):
        warnings.append(
            "La protección contra fuga de datos entre fuentes está desactivada."
        )

    return {
        "valid": len(errors) == 0,
        "project_name": project.get("name"),
        "declared_target_images": declared_total,
        "calculated_target_images": calculated_total,
        "number_of_classes": len(target_classes),
        "number_of_sources": len(sources),
        "enabled_sources": [
            source["id"]
            for source in sources
            if source.get("enabled", False)
        ],
        "automatic_sources": [
            source["id"]
            for source in sources
            if source.get("automatic_download", False)
        ],
        "errors": errors,
        "warnings": warnings,
    }


def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def print_report(report: dict[str, Any], output_path: Path) -> None:
    print()
    print("=" * 60)
    print("ARCHAEOSORT DATASET BUILDER - MANIFEST VALIDATION")
    print("=" * 60)
    print(f"Valid: {report['valid']}")
    print(
        "Target images: "
        f"{report.get('calculated_target_images', 0)}"
    )
    print(f"Classes: {report.get('number_of_classes', 0)}")
    print(f"Sources: {report.get('number_of_sources', 0)}")
    print(
        "Enabled sources: "
        f"{', '.join(report.get('enabled_sources', [])) or 'none'}"
    )

    if report["errors"]:
        print()
        print("ERRORS")
        for error in report["errors"]:
            print(f"- {error}")

    if report["warnings"]:
        print()
        print("WARNINGS")
        for warning in report["warnings"]:
            print(f"- {warning}")

    print()
    print(f"Report: {output_path.resolve()}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    try:
        manifest = load_manifest(args.config)
        report = validate_manifest(manifest)
        save_report(report, args.output)
        print_report(report, args.output)

    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
