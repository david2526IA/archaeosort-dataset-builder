from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from archaeosort_dataset_builder.config.settings import settings

app = typer.Typer(
    no_args_is_help=True,
    help="ArchaeoSort Dataset Builder - dataset engineering for computer vision.",
)


@app.callback()
def main(
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Path to the dataset directory."),
    ] = None,
    reports: Annotated[
        Path,
        typer.Option("--reports", help="Directory used to store reports."),
    ] = Path("reports"),
    outputs: Annotated[
        Path,
        typer.Option("--outputs", help="Directory used to store generated outputs."),
    ] = Path("outputs"),
) -> None:
    """Configure ArchaeoSort runtime paths."""
    if dataset is not None:
        settings.set_dataset(dataset)

    settings.set_reports(reports)
    settings.set_outputs(outputs)


@app.command("verify")
def verify_cmd() -> None:
    """Verify dataset structure and image integrity."""
    from archaeosort_dataset_builder.verify.verify import verify

    verify()


@app.command("analyze")
def analyze_cmd() -> None:
    """Analyze dataset image properties."""
    from archaeosort_dataset_builder.analyzer.analyzer import analyze

    analyze()


@app.command("duplicates")
def duplicates_cmd() -> None:
    """Detect duplicate and near-duplicate images."""
    from archaeosort_dataset_builder.duplicates.duplicates import duplicates

    duplicates()


@app.command("blur")
def blur_cmd() -> None:
    """Analyze image sharpness and blur."""
    from archaeosort_dataset_builder.blur.blur import blur

    blur()


@app.command("brightness")
def brightness_cmd() -> None:
    """Analyze image brightness."""
    from archaeosort_dataset_builder.brightness.brightness import brightness

    brightness()


@app.command("contrast")
def contrast_cmd() -> None:
    """Analyze image contrast."""
    from archaeosort_dataset_builder.contrast.contrast import contrast

    contrast()


@app.command("resolution")
def resolution_cmd() -> None:
    """Analyze image resolution."""
    from archaeosort_dataset_builder.resolution.resolution import resolution

    resolution()


@app.command("aspect")
def aspect_cmd() -> None:
    """Analyze image aspect ratios."""
    from archaeosort_dataset_builder.aspect_ratio.aspect_ratio import aspect_ratio

    aspect_ratio()


@app.command("classes")
def classes_cmd() -> None:
    """Analyze dataset class balance."""
    from archaeosort_dataset_builder.class_balance.class_balance import class_balance

    class_balance()


@app.command("quality")
def quality_cmd() -> None:
    """Calculate the dataset quality score."""
    from archaeosort_dataset_builder.quality.quality import quality

    quality()


@app.command("statistics")
def statistics_cmd() -> None:
    """Generate dataset statistics."""
    from archaeosort_dataset_builder.statistics.statistics import statistics

    statistics()


@app.command("report")
def report_cmd() -> None:
    """Generate the dataset report."""
    from archaeosort_dataset_builder.report.report import report

    report()


@app.command("embeddings")
def embeddings_cmd(
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Embedding batch size."),
    ] = 16,
) -> None:
    """Generate DINOv2 image embeddings."""
    from archaeosort_dataset_builder.embeddings.build import build_embeddings

    build_embeddings(batch_size=batch_size)


@app.command("index")
def index_cmd() -> None:
    """Build the FAISS similarity index."""
    from archaeosort_dataset_builder.embeddings.index import build_index

    build_index()


@app.command("search")
def search_cmd(
    image: Annotated[
        Path,
        typer.Argument(help="Query image."),
    ],
    k: Annotated[
        int,
        typer.Option("--k", min=1, help="Number of nearest neighbors."),
    ] = 5,
) -> None:
    """Search for visually similar images."""
    from archaeosort_dataset_builder.embeddings.search import search_similar

    search_similar(image, k=k)


@app.command("semantic-audit")
def semantic_audit_cmd(
    k: Annotated[
        int,
        typer.Option("--k", min=1, help="Number of semantic neighbors."),
    ] = 10,
) -> None:
    """Audit semantic consistency using embeddings."""
    from archaeosort_dataset_builder.embeddings.audit import semantic_audit

    semantic_audit(k=k)


@app.command("outliers")
def outliers_cmd(
    k: Annotated[
        int,
        typer.Option("--k", min=1, help="Number of neighbors."),
    ] = 10,
    percentile: Annotated[
        float,
        typer.Option(
            "--percentile",
            min=0.0,
            max=100.0,
            help="Outlier percentile threshold.",
        ),
    ] = 2.0,
) -> None:
    """Detect semantic visual outliers."""
    from archaeosort_dataset_builder.embeddings.outliers import detect_outliers

    detect_outliers(k=k, percentile=percentile)


@app.command("outlier-review")
def outlier_review_cmd(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum outliers to display."),
    ] = 30,
) -> None:
    """Generate a visual contact sheet for outlier review."""
    from archaeosort_dataset_builder.embeddings.outlier_review import (
        build_outlier_contact_sheet,
    )

    build_outlier_contact_sheet(limit=limit)


@app.command("pipeline")
def pipeline_cmd(
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Run the standard pipeline plus semantic analysis.",
        ),
    ] = False,
) -> None:
    """Run the complete dataset analysis pipeline."""
    from archaeosort_dataset_builder.pipeline.pipeline import run_pipeline

    run_pipeline()

    if not full:
        return

    from archaeosort_dataset_builder.embeddings.audit import semantic_audit
    from archaeosort_dataset_builder.embeddings.build import build_embeddings
    from archaeosort_dataset_builder.embeddings.index import build_index
    from archaeosort_dataset_builder.embeddings.outlier_review import (
        build_outlier_contact_sheet,
    )
    from archaeosort_dataset_builder.embeddings.outliers import detect_outliers

    embedding_file = settings.outputs / "embeddings" / "dinov2_embeddings.npy"
    paths_file = settings.outputs / "embeddings" / "dinov2_paths.json"

    if embedding_file.exists() and paths_file.exists():
        typer.echo("[CACHE] Existing DINOv2 embeddings found.")
    else:
        typer.echo("[BUILD] Generating DINOv2 embeddings...")
        build_embeddings(batch_size=16)

    build_index()
    semantic_audit(k=10)
    detect_outliers(k=10, percentile=2.0)
    build_outlier_contact_sheet(limit=30)


if __name__ == "__main__":
    app()
