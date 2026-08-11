from __future__ import annotations

from time import perf_counter

from archaeosort_dataset_builder.analyzer.analyzer import analyze
from archaeosort_dataset_builder.aspect_ratio.aspect_ratio import aspect_ratio
from archaeosort_dataset_builder.blur.blur import blur
from archaeosort_dataset_builder.brightness.brightness import brightness
from archaeosort_dataset_builder.class_balance.class_balance import class_balance
from archaeosort_dataset_builder.contrast.contrast import contrast
from archaeosort_dataset_builder.duplicates.duplicates import duplicates
from archaeosort_dataset_builder.quality.quality import quality
from archaeosort_dataset_builder.report.report import report
from archaeosort_dataset_builder.resolution.resolution import resolution
from archaeosort_dataset_builder.statistics.statistics import statistics
from archaeosort_dataset_builder.verify.verify import verify


def run_pipeline():

    steps = [
        ("verify", verify),
        ("analyze", analyze),
        ("duplicates", duplicates),
        ("blur", blur),
        ("brightness", brightness),
        ("contrast", contrast),
        ("resolution", resolution),
        ("aspect", aspect_ratio),
        ("classes", class_balance),
        ("quality", quality),
        ("statistics", statistics),
        ("report", report),
    ]

    print("=" * 60)
    print("ARCHAEOSORT DATASET PIPELINE")
    print("=" * 60)

    total_start = perf_counter()

    for name, function in steps:
        print()
        print("-" * 60)
        print(f"STEP: {name.upper()}")
        print("-" * 60)

        start = perf_counter()

        function()

        elapsed = perf_counter() - start

        print(f"[OK] {name} completed in {elapsed:.2f}s")

    total_elapsed = perf_counter() - total_start

    print()
    print("=" * 60)
    print("PIPELINE FINISHED")
    print("=" * 60)
    print(f"Steps       : {len(steps)}")
    print(f"Total time  : {total_elapsed:.2f}s")


if __name__ == "__main__":
    run_pipeline()
