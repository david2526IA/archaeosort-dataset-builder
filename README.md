# ArchaeoSort Dataset Builder

**English** | [Español](README_ES.md)

> Reproducible dataset construction, quality analysis and semantic auditing pipeline for archaeological computer vision.

ArchaeoSort Dataset Builder is a Python toolkit for building, validating and auditing image datasets before training computer vision models.

It is part of the **ArchaeoSort-AI** ecosystem and focuses on reproducible dataset engineering for archaeological computer vision.

## Features

- Dataset verification and validation
- Duplicate detection
- Blur analysis
- Brightness analysis
- Contrast analysis
- Resolution analysis
- Aspect-ratio analysis
- Class-balance analysis
- Dataset statistics
- Automated quality reports
- DINOv2 visual embeddings
- FAISS similarity indexing
- Semantic dataset auditing
- Visual outlier detection
- Human-review contact sheets
- Reproducible, versioned train/val/test dataset builds
- Object-isolation preprocessing (background/scale-bar removal)
- ResNet18 baseline classifier (transfer learning)
- Automated end-to-end pipeline
- Pytest test suite
- Ruff linting
- GitHub Actions CI

## Pipeline

```text
Dataset
   |
   v
Validation
   |
   v
Quality Analysis
   |
   +--> Duplicates
   +--> Blur
   +--> Brightness
   +--> Contrast
   +--> Resolution
   +--> Aspect Ratio
   +--> Class Balance
   +--> Statistics
   |
   v
DINOv2 Embeddings
   |
   v
FAISS Index
   |
   +--> Semantic Audit
   +--> Visual Outlier Detection
   |
   v
Reports / Human Review
```

## CLI

Show available commands:

```powershell
uv run --no-sync python builder.py --help
```

Available commands:

```text
verify
analyze
duplicates
quality
blur
brightness
resolution
contrast
aspect
classes
statistics
report
embeddings
index
search
semantic-audit
pipeline
outlier-review
outliers
```

Run the complete pipeline:

```powershell
uv run --no-sync python builder.py pipeline
```

## Build (reproducible, versioned dataset)

`archaeosort build` turns a class-folder source dataset into a versioned train/val/test dataset, reusing the existing (deterministic) splitter:

```powershell
uv run archaeosort --dataset <path-to-source-dataset> build --output <path-to-output> --train 0.70 --val 0.15 --test 0.15 --seed 42 [--copy] [--force]
```

- `--copy` copies files instead of hardlinking (hardlink with a copy fallback is the default).
- `--force` is required to overwrite a non-empty output directory; without it, `build` refuses to touch existing output.
- The source dataset and the output directory must be different paths — `build` refuses to run otherwise, since `--force` can delete the output directory.

Output layout:

```text
<output>/
|-- train/<class>/...
|-- val/<class>/...
|-- test/<class>/...
|-- split_report.json   # per-split, per-class image counts and the split mode
|-- provenance.json      # source dataset, seed, ratios, classes, timestamp, mode
`-- manifest.json         # file count, size and a deterministic SHA-256 content fingerprint
```

`manifest.json`'s `sha256` field is a content fingerprint: it depends only on the dataset's file paths and bytes (chunked-read; `manifest.json`, `split_report.json` and `provenance.json` are always excluded — they're build bookkeeping, not dataset content), never on wall-clock time, so two builds of identical content always produce the same fingerprint. `version`/`created` remain timestamp-based metadata.

`archaeosort build` is covered by its own test suite (`tests/test_dataset_builder.py`, `tests/test_cli_build.py`) plus the exporter/versioning hardening tests (`tests/test_dataset_exporter.py`, `tests/test_dataset_version.py`).

## Baseline Classifier (Fase G)

`archaeosort train-baseline` trains a ResNet18 transfer-learning classifier on a `build()`-shaped train/val/test dataset and evaluates it on the held-out test split:

```powershell
uv run archaeosort train-baseline --data <path-to-build-output> --output <path-to-run-dir> --epochs 15 --patience 3 [--batch-size 16] [--lr 0.001] [--freeze-backbone/--no-freeze-backbone] [--seed 42]
```

It writes `best_model.pt`, `training_history.json` and `evaluation.json` (a per-class `classification_report` plus a confusion matrix, evaluated on the test split) to `--output`, restoring the best checkpoint by validation accuracy before evaluating (early stopping via `--patience`). Uses CUDA automatically when available, falling back to CPU otherwise.

### Object cropping — fixing a shortcut-learning bug

A first training run on the raw `stone_figshare` source images hit a suspicious **100% test accuracy** after only two epochs. Duplicate-image leakage was ruled out first (`archaeosort duplicates` found 0 confirmed duplicate groups across 5,319 images), then visual inspection of sample images found the real cause: the two classes had been photographed with different, internally-consistent studio setups (different backgrounds, different scale-bar/label-card styles). The model was separating classes by photography style, not object content — a textbook dataset-bias / shortcut-learning failure, not a bug in the dataset-engineering pipeline itself.

`archaeosort crop-objects` removes that confound with classical computer vision (no ML, nothing to overfit): it thresholds the (mostly uniform) background, finds the object's contour — preferring contours that don't touch the frame border, since scale bars and label cards are usually placed at the edge while the specimen is normally centered — and crops to that bounding box:

```powershell
uv run archaeosort crop-objects --source <class-folder dataset> --output <path> [--margin 0.06] [--force]
```

It mirrors the source `<class>/` layout under `--output` and writes `crop_report.json` (per-class counts of cropped / low-confidence / unreadable images) so low-confidence crops can be audited later rather than trusted blindly.

Retraining on the cropped dataset produced a believable **95.9% test accuracy** (100% precision / 84.8% recall on `natural_rock`, 94.6% precision / 100% recall on `stone_artefact` — consistent with the 73%/27% class imbalance in the source data, not a shortcut), comfortably ahead of an earlier, smaller-dataset baseline that reached 83%.

`train-baseline` and `crop-objects` are covered by their own test suites (`tests/test_classifier_baseline_layout.py`, `tests/test_cli_train_baseline.py`, `tests/test_object_crop.py`, `tests/test_cli_crop_objects.py`).

## Semantic Analysis

The project combines traditional dataset-quality checks with modern visual representation learning.

### DINOv2

DINOv2 converts images into visual embeddings that capture semantic visual information.

### FAISS

FAISS provides efficient nearest-neighbour search over the generated embeddings.

Together they enable:

- Visual similarity search
- Semantic dataset exploration
- Potential label-error detection
- Outlier discovery
- Dataset auditing at scale

## Dataset Results

The pipeline has been tested on an archaeological image dataset containing **5,319 JPEG images**.

| Metric | Result |
| --- | ---: |
| Images analyzed | 5,319 |
| Confirmed duplicate groups | 0 |
| Sharp images | 3,537 |
| Blurry images | 1,782 |
| Low contrast | 669 |
| Normal contrast | 4,635 |
| High contrast | 15 |
| Dark images | 1 |
| Normal brightness | 2,794 |
| Bright images | 2,524 |
| Visual outliers | 107 |

These are dataset-analysis results and not model-performance metrics.

## Reports

The pipeline produces JSON, HTML and visual review artifacts under `reports/`.

Examples:

```text
reports/
|-- blur.json
|-- brightness.json
|-- contrast.json
|-- duplicates.json
|-- resolution.json
|-- class_balance.json
|-- semantic_audit.json
|-- outliers.json
|-- verify_report.json
|-- report.html
|-- outlier_review/
|   `-- outliers_top30.jpg
`-- semantic_review/
    `-- suspicious review images
```

### Visual Outlier Review

![Visual outlier review](reports/outlier_review/outliers_top30.jpg)

### Semantic Review

![Semantic review](reports/semantic_review/suspicious_001_full.jpg)

## Installation

Requirements:

- Python >=3.11,<3.13
- Git
- uv

Install dependencies:

```powershell
uv sync
```

## Development

Run linting:

```powershell
uv run --no-sync ruff check .
```

Run tests:

```powershell
uv run --no-sync pytest -q
```

Current test status:

```text
26 passed
```

Run coverage:

```powershell
uv run --no-sync pytest --cov=archaeosort_dataset_builder --cov-report=term-missing
```

Current baseline coverage: **21%**.

## Continuous Integration

GitHub Actions CI is configured in `.github/workflows/ci.yml`.

The workflow validates code quality and executes the automated test suite on repository changes.

## Project Structure

```text
archaeosort-dataset-builder/
|-- builder.py
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- src/
|   `-- archaeosort_dataset_builder/
|       |-- analyzer/
|       |-- aspect_ratio/
|       |-- blur/
|       |-- brightness/
|       |-- builder/          # archaeosort build (versioned train/val/test datasets)
|       |-- class_balance/
|       |-- classifier/       # archaeosort train-baseline (ResNet18 classifier)
|       |-- config/
|       |-- contrast/
|       |-- duplicates/
|       |-- embeddings/
|       |-- exporter/
|       |-- pipeline/
|       |-- preprocessing/    # archaeosort crop-objects (background/scale-bar removal)
|       |-- quality/
|       |-- report/
|       |-- resolution/
|       |-- splitter/
|       |-- statistics/
|       |-- validator/
|       |-- verify/
|       `-- versioning/
|-- tests/
|-- reports/
`-- .github/workflows/
```

## Documentation

Additional project documentation lives under [`docs/`](docs/):

- [`ROADMAP_ARCHAEOSORT.md`](docs/ROADMAP_ARCHAEOSORT.md) — phased roadmap for the broader ArchaeoSort-AI system (dataset engineering, classifier, inference API, frontend, Isaac Sim).
- [`ARCHAEOSORT_AGENT_CONTEXT.md`](docs/ARCHAEOSORT_AGENT_CONTEXT.md) — technical handoff notes for AI coding assistants working on this repo.
- A Spanish-language technical memo (`.docx`) covering the project in more depth.

## Engineering Goals

- Reproducible dataset construction
- Modular architecture
- Automated quality validation
- Dataset quality before model training
- Human-in-the-loop review
- Semantic computer-vision analysis
- Automated testing
- Continuous integration

## Roadmap

- Inference API (FastAPI) serving the baseline classifier
- Frontend for reviewing predictions (Next.js)
- Docker support and CI/CD for the full stack
- Active-learning integration
- Interactive dataset dashboard
- Improved semantic duplicate detection
- Dataset version comparison
- Additional embedding backends
- Isaac Sim: synthetic training data + camera/conveyor-belt simulation
- Integration with the complete ArchaeoSort-AI system

## Status

**Version: 0.1.0**

- Quality analysis: implemented
- Duplicate detection: implemented
- DINOv2 embeddings: implemented
- FAISS indexing: implemented
- Semantic auditing: implemented
- Visual outlier detection: implemented
- Pipeline orchestration: implemented
- Reproducible dataset builds: implemented
- Object-isolation preprocessing: implemented
- Baseline classifier (ResNet18, 95.9% test accuracy on the cropped dataset): implemented
- CI: configured

## Author

**David Falcon Perez**

Automation, robotics, artificial intelligence and applied computer vision.

---

ArchaeoSort Dataset Builder is part of the broader **ArchaeoSort-AI** project.

