import sys
from pathlib import Path

import pytest

from archaeosort_dataset_builder.exporter.dataset_exporter import ExportError, export_dataset, main


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


def test_export_refuses_non_empty_output_without_force(tmp_path: Path, monkeypatch):
    # Fase E hardening: exporting into a non-empty output directory must now
    # be refused unless --force / force=True is given explicitly (previously
    # it deleted the existing output unconditionally).
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

    with pytest.raises(FileExistsError):
        main()

    assert (output_dir / "old.txt").exists()


def test_export_replaces_existing_output_with_force(tmp_path: Path, monkeypatch):
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
            "--force",
        ],
    )

    main()

    assert (output_dir / "new.txt").exists()
    assert not (output_dir / "old.txt").exists()


def test_export_dataset_rejects_input_equal_to_output(tmp_path: Path):
    same_dir = tmp_path / "same"
    same_dir.mkdir()

    with pytest.raises(ExportError):
        export_dataset(same_dir, same_dir)


def test_export_dataset_callable_api_returns_resolved_output(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a", encoding="utf8")

    result = export_dataset(input_dir, output_dir)

    assert result == output_dir.resolve()
    assert (output_dir / "a.txt").exists()


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
