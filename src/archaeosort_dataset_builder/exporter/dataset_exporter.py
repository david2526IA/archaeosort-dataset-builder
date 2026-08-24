"""Copy (export) a dataset directory tree.

Fase E hardening: the previous implementation deleted an existing output
directory unconditionally, with no way to opt out. It is now safe by
default (refuses a non-empty output unless ``force=True`` / ``--force``,
matching the convention already used by
``splitter.dataset_splitter.prepare_output_directory``), and exposes a
callable API (``export_dataset``) instead of only an argparse script, so
other code (e.g. a future build/versioning integration) can call it
directly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class ExportError(ValueError):
    """Raised when the export configuration or inputs are invalid."""


def export_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    force: bool = False,
) -> Path:
    """Copy ``input_dir`` to ``output_dir``.

    Safe by default: raises ``FileExistsError`` if ``output_dir`` already
    exists and is not empty, unless ``force=True`` (in which case the
    existing output is removed first). Raises ``ExportError`` if the
    resolved input and output paths are the same directory.

    Returns the resolved output directory.
    """
    resolved_input = Path(input_dir).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()

    if not resolved_input.exists():
        raise FileNotFoundError(f"Input directory not found: {resolved_input}")

    if not resolved_input.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {resolved_input}")

    if resolved_input == resolved_output:
        raise ExportError(
            "Output directory must be different from the input directory "
            f"(both resolved to {resolved_input})."
        )

    if resolved_output.exists() and any(resolved_output.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory already contains files: {resolved_output}. "
                "Pass force=True (or --force) to overwrite it."
            )
        shutil.rmtree(resolved_output)

    shutil.copytree(resolved_input, resolved_output, dirs_exist_ok=True)

    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a dataset directory tree.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = export_dataset(
        args.input_dir,
        args.output_dir,
        force=args.force,
    )

    print("=" * 60)
    print("ARCHAEOSORT DATASET EXPORTER")
    print("=" * 60)
    print(f"Input : {args.input_dir}")
    print(f"Output: {output_dir}")
    print("Export completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
