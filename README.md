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
|       |-- blur/
|       |-- brightness/
|       |-- contrast/
|       |-- duplicates/
|       |-- embeddings/
|       |-- pipeline/
|       |-- quality/
|       |-- report/
|       |-- resolution/
|       |-- statistics/
|       |-- validator/
|       `-- verify/
|-- tests/
|-- reports/
`-- .github/workflows/
```

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

- Increase automated test coverage
- Configurable pipeline parameters
- Interactive dataset dashboard
- Improved semantic duplicate detection
- Dataset version comparison
- Active-learning integration
- Additional embedding backends
- Docker support
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
- Tests: 26 passing
- CI: configured

## Author

**David Falcon Perez**

Automation, robotics, artificial intelligence and applied computer vision.

---

ArchaeoSort Dataset Builder is part of the broader **ArchaeoSort-AI** project.

