from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    dataset: Path = Path(
        "C:/Users/farqu/Desktop/proyectos/archaeosort-datasets/datasets/v2/stone_figshare"
    )

    reports: Path = Path("reports")

    outputs: Path = Path("outputs")

    image_extensions: tuple = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".avif",
    )

    def set_dataset(self, path):

        self.dataset = Path(path)


settings = Settings()
