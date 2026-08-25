from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from archaeosort_dataset_builder.cli import app

runner = CliRunner()


def _make_source_dataset(root: Path) -> Path:
    source = root / "source"
    for class_name in ("natural_rock", "stone_artefact"):
        class_dir = source / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            img = np.full((200, 200, 3), 230, dtype=np.uint8)
            cv2.circle(img, (100, 100), 40, (50, 50, 50), thickness=-1)
            cv2.imwrite(str(class_dir / f"img_{index:02d}.jpg"), img)
    return source


def test_crop_objects_help_lists_expected_options():
    result = runner.invoke(app, ["crop-objects", "--help"])
    assert result.exit_code == 0
    for flag in ("--source", "--output", "--margin", "--force"):
        assert flag in result.stdout


def test_cli_crop_objects_writes_mirrored_dataset_and_report(tmp_path):
    source = _make_source_dataset(tmp_path)
    output = tmp_path / "cropped"

    result = runner.invoke(app, ["crop-objects", "--source", str(source), "--output", str(output)])

    assert result.exit_code == 0, result.stdout
    assert (output / "crop_report.json").exists()
    assert (output / "natural_rock").is_dir()
    assert (output / "stone_artefact").is_dir()
    assert len(list((output / "natural_rock").glob("*.jpg"))) == 3


def test_cli_crop_objects_requires_source_and_output():
    result = runner.invoke(app, ["crop-objects"])
    assert result.exit_code != 0
