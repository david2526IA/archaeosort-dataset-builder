from pathlib import Path

from PIL import Image

from archaeosort_dataset_builder.class_balance.class_balance import (
    compute_class_balance,
)


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100)).save(path)


def test_class_folder_layout(tmp_path):
    create_image(tmp_path / "natural_rock" / "rock1.jpg")
    create_image(tmp_path / "natural_rock" / "rock2.jpg")
    create_image(tmp_path / "stone_artefact" / "artifact1.jpg")

    result = compute_class_balance(tmp_path)

    assert result["layout"] == "class_folders"
    assert result["total"] == 3
    assert result["classes"]["natural_rock"] == 2
    assert result["classes"]["stone_artefact"] == 1


def test_split_layout(tmp_path):
    create_image(tmp_path / "train" / "natural_rock" / "rock1.jpg")
    create_image(tmp_path / "train" / "stone_artefact" / "artifact1.jpg")
    create_image(tmp_path / "val" / "natural_rock" / "rock2.jpg")
    create_image(tmp_path / "test" / "stone_artefact" / "artifact2.jpg")

    result = compute_class_balance(tmp_path)

    assert result["layout"] == "split"
    assert result["total"] == 4
    assert result["classes"]["natural_rock"] == 2
    assert result["classes"]["stone_artefact"] == 2


def test_non_image_files_are_ignored(tmp_path):
    create_image(tmp_path / "natural_rock" / "rock.jpg")

    metadata = tmp_path / "natural_rock" / "metadata.txt"
    metadata.write_text("not an image", encoding="utf8")

    result = compute_class_balance(tmp_path)

    assert result["total"] == 1
    assert result["classes"]["natural_rock"] == 1


def test_missing_dataset_raises_error(tmp_path):
    missing = tmp_path / "does_not_exist"

    try:
        compute_class_balance(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
