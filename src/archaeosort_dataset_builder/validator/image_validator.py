from pathlib import Path

from PIL import Image


def validate_image(image_path: Path) -> list[str]:
    issues: list[str] = []

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            if image.width < 64 or image.height < 64:
                issues.append("Image resolution is too small.")

            if image.mode != "RGB":
                issues.append(f"Image mode is {image.mode} instead of RGB.")

    except (OSError, ValueError) as exc:
        issues.append(f"Cannot open image: {exc}")

    return issues
