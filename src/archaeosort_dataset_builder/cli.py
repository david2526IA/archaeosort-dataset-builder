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


@app.command("build")
def build_cmd(
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory where the versioned dataset will be written."),
    ],
    train: Annotated[
        float,
        typer.Option("--train", help="Train split ratio."),
    ] = 0.70,
    val: Annotated[
        float,
        typer.Option("--val", help="Validation split ratio."),
    ] = 0.15,
    test: Annotated[
        float,
        typer.Option("--test", help="Test split ratio."),
    ] = 0.15,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for the deterministic split."),
    ] = 42,
    copy: Annotated[
        bool,
        typer.Option("--copy", help="Copy files instead of hardlinking."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite a non-empty output directory."),
    ] = False,
) -> None:
    """Build a reproducible, versioned train/val/test dataset."""
    from archaeosort_dataset_builder.builder.dataset_builder import build_dataset

    source = settings.require_dataset()

    result = build_dataset(
        dataset=source,
        output=output,
        train_ratio=train,
        val_ratio=val,
        test_ratio=test,
        seed=seed,
        copy_files=copy,
        force=force,
    )

    typer.echo(f"Dataset built at: {result['output_dir']}")
    typer.echo(f"Split report: {result['split_report_path']}")
    typer.echo(f"Provenance:   {result['provenance_path']}")
    typer.echo(f"Manifest:     {result['manifest_path']}")


@app.command("crop-objects")
def crop_objects_cmd(
    source: Annotated[
        Path,
        typer.Option("--source", help="Class-folder dataset to crop (e.g. the raw source used by 'build')."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory to write the cropped, class-folder-mirrored dataset to."),
    ],
    margin: Annotated[
        float,
        typer.Option("--margin", min=0.0, help="Extra margin around the detected object, as a fraction of its box."),
    ] = 0.06,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite a non-empty output directory."),
    ] = False,
) -> None:
    """Crop each image down to just the object, removing studio background
    and scale-bar/label cards (classical CV, no ML). Use this before 'build'
    if a duplicates/quality check didn't explain a suspiciously high
    classifier accuracy — background/scale-bar style can otherwise leak the
    class label to the model."""
    from archaeosort_dataset_builder.preprocessing.object_crop import crop_dataset

    report = crop_dataset(source=source, output=output, margin_frac=margin, force=force)

    typer.echo(f"Cropped dataset: {report['output']}")
    typer.echo(f"Report:          {report['report_path']}")
    typer.echo(f"Images: {report['total']['images']}  Cropped: {report['total']['cropped']}  "
               f"Low-confidence: {report['total']['low_confidence']}  Unreadable: {report['total']['unreadable']}")
    for class_name, stats in report["classes"].items():
        typer.echo(
            f"  {class_name}: {stats['images']} images, {stats['cropped']} cropped, "
            f"{stats['low_confidence']} low-confidence, {stats['unreadable']} unreadable"
        )


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


@app.command("train-baseline")
def train_baseline_cmd(
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            help="Path to a train/val/test dataset, as produced by 'archaeosort build'.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory to write the model and metrics to."),
    ],
    epochs: Annotated[
        int,
        typer.Option("--epochs", min=1, help="Maximum number of training epochs."),
    ] = 15,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Training batch size."),
    ] = 16,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate."),
    ] = 1e-3,
    freeze_backbone: Annotated[
        bool,
        typer.Option(
            "--freeze-backbone/--no-freeze-backbone",
            help="Freeze the pretrained ResNet18 backbone (only train the final layer).",
        ),
    ] = False,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            min=1,
            help="Stop training after this many epochs without validation improvement.",
        ),
    ] = 3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed."),
    ] = 42,
) -> None:
    """Train a ResNet18 baseline classifier and evaluate it on the test split."""
    from archaeosort_dataset_builder.classifier.baseline import TrainConfig, train_baseline

    result = train_baseline(
        TrainConfig(
            data_dir=data,
            output_dir=output,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            freeze_backbone=freeze_backbone,
            patience=patience,
            seed=seed,
        )
    )

    evaluation = result["evaluation"]
    typer.echo(f"Model:      {result['model_path']}")
    typer.echo(f"History:    {result['history_path']}")
    typer.echo(f"Evaluation: {result['evaluation_path']}")
    typer.echo(
        f"Best val accuracy: {evaluation['best_validation_accuracy']:.4f}  "
        f"Test accuracy: {evaluation['test_accuracy']:.4f}  "
        f"Epochs run: {evaluation['epochs_run']}/{evaluation['epochs_configured']}"
        f"{' (stopped early)' if evaluation['stopped_early'] else ''}"
    )


if __name__ == "__main__":
    app()
