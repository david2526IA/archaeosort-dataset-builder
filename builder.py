import argparse
from pathlib import Path

from archaeosort_dataset_builder.analyzer.analyzer import analyze
from archaeosort_dataset_builder.aspect_ratio.aspect_ratio import aspect_ratio
from archaeosort_dataset_builder.blur.blur import blur
from archaeosort_dataset_builder.brightness.brightness import brightness
from archaeosort_dataset_builder.class_balance.class_balance import class_balance
from archaeosort_dataset_builder.config.settings import settings
from archaeosort_dataset_builder.contrast.contrast import contrast
from archaeosort_dataset_builder.duplicates.duplicates import duplicates
from archaeosort_dataset_builder.embeddings.audit import semantic_audit
from archaeosort_dataset_builder.embeddings.build import build_embeddings
from archaeosort_dataset_builder.embeddings.index import build_index
from archaeosort_dataset_builder.embeddings.outlier_review import build_outlier_contact_sheet
from archaeosort_dataset_builder.embeddings.outliers import detect_outliers
from archaeosort_dataset_builder.embeddings.search import search_similar
from archaeosort_dataset_builder.pipeline.pipeline import run_pipeline
from archaeosort_dataset_builder.quality.quality import quality
from archaeosort_dataset_builder.report.report import report
from archaeosort_dataset_builder.resolution.resolution import resolution
from archaeosort_dataset_builder.statistics.statistics import statistics
from archaeosort_dataset_builder.verify.verify import verify

parser = argparse.ArgumentParser(description="ArchaeoSort Dataset Builder")

parser.add_argument(
    "--dataset",
    type=Path,
    help="Path to the dataset directory.",
)
parser.add_argument(
    "--reports",
    type=Path,
    default=Path("reports"),
    help="Directory used to store reports.",
)
parser.add_argument(
    "--outputs",
    type=Path,
    default=Path("outputs"),
    help="Directory used to store generated outputs.",
)

sub = parser.add_subparsers(dest="command")

for cmd in [
    "verify",
    "analyze",
    "duplicates",
    "quality",
    "blur",
    "brightness",
    "resolution",
    "contrast",
    "aspect",
    "classes",
    "statistics",
    "report",
]:
    sub.add_parser(cmd)


embeddings_parser = sub.add_parser("embeddings")
embeddings_parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
)

sub.add_parser("index")


search_parser = sub.add_parser("search")
search_parser.add_argument("image", type=Path)
search_parser.add_argument("--k", type=int, default=5)


audit_parser = sub.add_parser("semantic-audit")
audit_parser.add_argument("--k", type=int, default=10)


pipeline_parser = sub.add_parser("pipeline")
pipeline_parser.add_argument("--full", action="store_true")

outlier_review_parser = sub.add_parser("outlier-review")
outlier_review_parser.add_argument("--limit", type=int, default=30)

outliers_parser = sub.add_parser("outliers")
outliers_parser.add_argument("--k", type=int, default=10)
outliers_parser.add_argument("--percentile", type=float, default=2.0)


args = parser.parse_args()

if args.dataset is not None:
    settings.set_dataset(args.dataset)

settings.set_reports(args.reports)
settings.set_outputs(args.outputs)


match args.command:
    case "verify":
        verify()

    case "analyze":
        analyze()

    case "duplicates":
        duplicates()

    case "quality":
        quality()

    case "blur":
        blur()

    case "brightness":
        brightness()

    case "resolution":
        resolution()

    case "contrast":
        contrast()

    case "aspect":
        aspect_ratio()

    case "classes":
        class_balance()

    case "statistics":
        statistics()

    case "report":
        report()

    case "embeddings":
        build_embeddings(batch_size=args.batch_size)

    case "index":
        build_index()

    case "search":
        search_similar(args.image, args.k)

    case "semantic-audit":
        semantic_audit(k=args.k)

    case "pipeline":
        run_pipeline()

        if args.full:
            embedding_file = Path("outputs/embeddings/dinov2_embeddings.npy")
            paths_file = Path("outputs/embeddings/dinov2_paths.json")

            if embedding_file.exists() and paths_file.exists():
                print("[CACHE] Existing DINOv2 embeddings found.")
            else:
                print("[BUILD] Generating DINOv2 embeddings...")
                build_embeddings(batch_size=16)

            build_index()
            semantic_audit(k=10)
            detect_outliers(k=10, percentile=2.0)
            build_outlier_contact_sheet(limit=30)

    case "outlier-review":
        build_outlier_contact_sheet(limit=args.limit)

    case "outliers":
        detect_outliers(k=args.k, percentile=args.percentile)

    case _:
        parser.print_help()




