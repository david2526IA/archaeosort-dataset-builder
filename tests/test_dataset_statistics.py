from pathlib import Path

from archaeosort_dataset_builder.statistics.dataset_statistics import compute_statistics


def test_compute_statistics(tmp_path: Path):

    artifact = tmp_path / "artifact"
    rock = tmp_path / "natural_rock"

    artifact.mkdir()
    rock.mkdir()

    (artifact / "image1.jpg").touch()
    (artifact / "image2.png").touch()
    (rock / "image3.jpg").touch()

    statistics = compute_statistics(tmp_path)

    assert statistics["total_images"] == 3

    assert statistics["classes"]["artifact"] == 2
    assert statistics["classes"]["natural_rock"] == 1

    assert statistics["extensions"][".jpg"] == 2
    assert statistics["extensions"][".png"] == 1


def test_ignore_non_images(tmp_path: Path):

    dataset = tmp_path / "artifact"
    dataset.mkdir()

    (dataset / "image.jpg").touch()
    (dataset / "notes.txt").touch()

    statistics = compute_statistics(tmp_path)

    assert statistics["total_images"] == 1
