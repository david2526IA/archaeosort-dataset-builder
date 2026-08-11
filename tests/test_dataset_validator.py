from pathlib import Path

from PIL import Image

from archaeosort_dataset_builder.validator.class_validator import validate_classes
from archaeosort_dataset_builder.validator.dataset_validator import validate_dataset
from archaeosort_dataset_builder.validator.image_validator import validate_image
from archaeosort_dataset_builder.validator.metadata_validator import validate_metadata


def test_missing_metadata(tmp_path: Path):
    issues = validate_metadata(tmp_path)

    assert "Missing metadata file: README.md" in issues
    assert "Missing metadata file: LICENSE" in issues


def test_valid_metadata(tmp_path: Path):
    (tmp_path / "README.md").touch()
    (tmp_path / "LICENSE").touch()

    issues = validate_metadata(tmp_path)

    assert issues == []


def test_dataset_without_classes(tmp_path: Path):
    issues = validate_classes(tmp_path)

    assert "Dataset has no class folders." in issues


def test_empty_class(tmp_path: Path):
    class_dir = tmp_path / "stone_artefact"
    class_dir.mkdir()

    issues = validate_classes(tmp_path)

    assert "Class 'stone_artefact' is empty." in issues


def test_valid_rgb_image(tmp_path: Path):
    image_path = tmp_path / "image.jpg"

    image = Image.new("RGB", (128, 128))
    image.save(image_path)

    issues = validate_image(image_path)

    assert issues == []


def test_small_image_detected(tmp_path: Path):
    image_path = tmp_path / "small.jpg"

    image = Image.new("RGB", (32, 32))
    image.save(image_path)

    issues = validate_image(image_path)

    assert "Image resolution is too small." in issues


def test_non_rgb_image_detected(tmp_path: Path):
    image_path = tmp_path / "gray.png"

    image = Image.new("L", (128, 128))
    image.save(image_path)

    issues = validate_image(image_path)

    assert "Image mode is L instead of RGB." in issues


def test_corrupted_image_detected(tmp_path: Path):
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"this is not an image")

    issues = validate_image(image_path)

    assert any("Cannot open image:" in issue for issue in issues)


def test_complete_valid_dataset(tmp_path: Path):
    (tmp_path / "README.md").touch()
    (tmp_path / "LICENSE").touch()

    class_dir = tmp_path / "stone_artefact"
    class_dir.mkdir()

    image_path = class_dir / "artifact.jpg"

    image = Image.new("RGB", (128, 128))
    image.save(image_path)

    issues = validate_dataset(tmp_path)

    assert issues == []
