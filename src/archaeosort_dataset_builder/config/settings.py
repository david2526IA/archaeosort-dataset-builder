from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime configuration for ArchaeoSort Dataset Builder."""

    dataset: Path | None = None
    reports: Path = Path("reports")
    outputs: Path = Path("outputs")

    image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".avif",
    )

    def set_dataset(self, path: str | Path) -> None:
        self.dataset = Path(path).expanduser().resolve()

    def set_reports(self, path: str | Path) -> None:
        self.reports = Path(path).expanduser().resolve()

    def set_outputs(self, path: str | Path) -> None:
        self.outputs = Path(path).expanduser().resolve()

    def require_dataset(self) -> Path:
        if self.dataset is None:
            raise ValueError(
                "No dataset configured. Use --dataset <path> to specify a dataset."
            )

        if not self.dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset}")

        if not self.dataset.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.dataset}")

        return self.dataset


settings = Settings()
