
from PIL import Image

from archaeosort_dataset_builder.config.settings import settings
from archaeosort_dataset_builder.quality.quality_scan import scan_quality


def test_quality_scan_creates_reports(tmp_path):
    dataset = tmp_path / "dataset"
    reports = tmp_path / "reports"

    class_dir = dataset / "class_a"
    class_dir.mkdir(parents=True)

    Image.new("RGB", (128, 128), (20, 20, 20)).save(class_dir / "dark.jpg")
    Image.new("RGB", (128, 128), (220, 220, 220)).save(class_dir / "bright.jpg")

    old_reports = settings.reports

    try:
        settings.reports = reports

        result = scan_quality(
            dataset=dataset,
            max_side=128,
        )

        assert result["images"] == 2
        assert result["brightness"]["dark"] == 1
        assert result["brightness"]["bright"] == 1

        assert (reports / "blur.json").exists()
        assert (reports / "brightness.json").exists()
        assert (reports / "contrast.json").exists()

    finally:
        settings.reports = old_reports
