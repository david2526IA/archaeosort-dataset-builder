import sys
from pathlib import Path

import pytest

from archaeosort_dataset_builder.exporter.dataset_exporter import main


def test_export_dataset(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    artifact_dir = input_dir / "stone_artefact"
    rock_dir = input_dir / "natural_rock"

    artifact_dir.mkdir(parents=True)
    rock_dir.mkdir(parents=True)

    (artifact_dir / "artifact.jpg").write_bytes(b"artifact")
    (rock_dir / "rock.jpg").write_bytes(b"rock")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_exporter",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    assert output_dir.exists()
    assert (output_dir / "stone_artefact" / "artifact.jpg").exists()
    assert (output_dir / "natural_rock" / "rock.jpg").exists()


def test_export_replaces_existing_output(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "new.txt").write_text("new", encoding="utf8")
    (output_dir / "old.txt").write_text("old", encoding="utf8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_exporter",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    assert (output_dir / "new.txt").exists()
    assert not (output_dir / "old.txt").exists()


def test_export_missing_input_raises_error(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "does_not_exist"
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_exporter",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(FileNotFoundError):
        main()
