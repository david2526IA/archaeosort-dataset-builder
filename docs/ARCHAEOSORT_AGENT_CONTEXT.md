# ArchaeoSort Dataset Builder — AI Agent Handoff

Status: 2026-08-24

## Purpose

This file is the canonical handoff for an AI coding agent continuing the existing `archaeosort-dataset-builder` repository. It contains verified state, historical decisions, known errors, current architecture, real dataset results, and the next development milestone.

IMPORTANT: do not assume that a proposed code block from a prior chat was actually executed. Inspect the repository first and compare reality against this document.

## Operating rules for the next agent

1. Continue the existing repository; do not rebuild it from scratch.
2. Before editing, run Git status, CLI help, Ruff and Pytest.
3. Verify whether the proposed `builder/dataset_builder.py` exists and whether `build` is already in the CLI.
4. Preserve working behavior and existing tests.
5. Prefer small, testable changes.
6. Do not hardcode the developer's local dataset path in library defaults.
7. Use `uv` for the project environment and package execution.
8. Public CLI is Typer. Some old internal modules still use argparse; refactor gradually.
9. Keep datasets, embeddings, indexes, caches, generated reports and large outputs out of Git.
10. Every substantial feature should include tests and documentation.
11. Primary shell is Windows PowerShell.
12. The user prefers exact terminal commands, step-by-step execution, expected results, and documentation of decisions/errors/solutions.
13. Do not perform destructive overwrite silently.
14. Do not automatically push unless explicitly requested in the active workflow.
15. Immediate milestone: a safe, reproducible, versioned `archaeosort build`.

## Project identity

Repository: `archaeosort-dataset-builder`

Local repository:
`C:\Users\farqu\Desktop\proyectos\archaeosort-dataset-builder`

Broader project: ArchaeoSort-AI.

Python package: `archaeosort_dataset_builder`

Observed package version: `0.1.0`

Description:
`Reproducible dataset construction pipeline for archaeological computer vision.`

Python: `>=3.11,<3.13`

Tooling:
- Windows / PowerShell
- Python 3.12
- uv
- Git / GitHub
- Ruff
- Pytest
- Typer
- Pydantic
- Rich
- Pillow / OpenCV / NumPy
- Torch / Torchvision
- Transformers / timm
- FAISS

The repository is intended as a professional portfolio project demonstrating dataset engineering, data-centric computer vision, semantic image analysis, reproducibility, testing, packaging, CLI design and MLOps-oriented practices.

## Product goal

ArchaeoSort Dataset Builder should turn source image collections into auditable, reproducible ML datasets.

Target lifecycle:

```text
SOURCE DATASET
      |
      v
integrity verification
      |
      v
technical quality analysis
      |
      v
duplicates / balance / statistics
      |
      v
DINOv2 embeddings
      |
      v
FAISS similarity / semantic audit / outliers
      |
      v
HTML + JSON reports
      |
      v
reproducible BUILD
      |
      +--> train/
      +--> val/
      +--> test/
      +--> split_report.json
      +--> provenance.json
      +--> manifest.json
      +--> SHA-256 fingerprint
      |
      v
VERSIONED ML DATASET
```

## Real dataset used during development

Confirmed path:
`C:\Users\farqu\Desktop\proyectos\archaeosort-datasets\datasets\v2\stone_figshare`

Confirmed layout:

```text
stone_figshare/
├── natural_rock/
├── stone_artefact/
└── manifest.json
```

This is a root-level class-folder dataset, not a pre-existing train/val/test layout.

Confirmed class distribution:
- natural_rock: 1,438 images — 27.04%
- stone_artefact: 3,881 images — 72.96%
- total: 5,319 images

Confirmed image analysis:
- 5,319 JPEG images
- average width: 4597.7
- average height: 3091.8
- min width: 1024
- max width: 6000
- min height: 768
- max height: 4000

Never make this path the package default. It is only a local source/smoke-test dataset.

## Confirmed repository structure

```text
src/archaeosort_dataset_builder/
├── analyzer/
├── aspect_ratio/
├── blur/
├── brightness/
├── class_balance/
├── config/
├── contrast/
├── downloaders/
├── duplicates/
├── embeddings/
├── exporter/
├── pipeline/
├── processors/
├── quality/
├── report/
├── reports/
├── resolution/
├── splitter/
├── statistics/
├── validator/
├── verify/
├── versioning/
└── cli.py
```

Repository-level components known:
- `.github/workflows/ci.yml`
- `README.md`
- `README_ES.md`
- `LICENSE`
- `pyproject.toml`
- `uv.lock`
- legacy/root `builder.py`
- `tests/`

Potentially proposed but NOT confirmed:
`src/archaeosort_dataset_builder/builder/dataset_builder.py`

Verify before creating/replacing it.

## Runtime settings

The project originally used a hardcoded local dataset path. This was corrected.

Confirmed design of `config/settings.py`:
- `dataset: Path | None = None`
- `reports: Path = Path("reports")`
- `outputs: Path = Path("outputs")`
- supported extensions include jpg/jpeg/png/bmp/tif/tiff/webp/avif
- `set_dataset`, `set_reports`, `set_outputs`
- `require_dataset()` raises a useful error if dataset is missing, nonexistent or not a directory.

Design rule: functions should accept explicit paths or use runtime settings; no developer-specific absolute defaults.

## CLI and packaging

A Typer CLI was made installable.

Confirmed `pyproject.toml` concepts:

```toml
[project.scripts]
archaeosort = "archaeosort_dataset_builder.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/archaeosort_dataset_builder"]
```

Confirmed command:
`uv run archaeosort --help`

Confirmed global options:
- `--dataset <path>`
- `--reports <path>` default reports
- `--outputs <path>` default outputs

Confirmed commands at last check:
- verify
- analyze
- duplicates
- blur
- brightness
- contrast
- resolution
- aspect
- classes
- quality
- statistics
- report
- embeddings
- index
- search
- semantic-audit
- outliers
- outlier-review
- pipeline

No `build` command was confirmed at that point.

### CLI historical issues

Initially `--dataset` was not recognized. Runtime settings and global CLI options were added.

Plain `python builder.py` produced package import errors in some contexts; `uv run ...` is the preferred execution path.

After packaging, `archaeosort` initially was not found. Hatchling/project scripts plus `uv sync` fixed installation.

At one point `uv run archaeosort --help` raised:
`OpenBLAS error: Memory allocation still failed after 10 retries, giving up.`

Direct package imports and `python -m archaeosort_dataset_builder.cli --help` worked. The CLI was subsequently refactored and final `uv run archaeosort --help` is confirmed working. Avoid eager imports of heavy ML stacks for basic CLI startup.

Ruff B008 objected to Typer Option/Argument calls in defaults. The CLI moved toward `typing.Annotated`. A mechanical edit briefly caused a SyntaxError; the file was fixed. Compile after structural CLI edits.

## Core analysis modules

### Verify
Checks dataset structure, images, empty files and corruption.

Confirmed real result:
- directories: 2
- images: 5319
- empty files: 0
- corrupted images: 0
- .jpg: 5319

Generates `verify_report.json`.

### Analyze
Confirmed:
- images: 5319
- avg width 4597.7
- avg height 3091.8
- min/max width 1024/6000
- min/max height 768/4000
- JPEG 5319

### Duplicates
Confirmed:
- candidate groups: 7
- candidate images: 14
- md5 groups: 0
- sha256 groups: 0
- phash groups: 0

Generates `duplicates.json`.

### Blur
Uses grayscale and Laplacian variance, with reduced image size to avoid unnecessary full-resolution work.

Observed signature:
`blur(dataset=None, threshold=100, max_side=1280)`

Confirmed:
- sharp: 3537
- blurry: 1782
- unreadable: 0

### Brightness
Observed original thresholds:
- dark < 60
- bright > 190
- otherwise normal

Confirmed:
- dark 1
- normal 2794
- bright 2524

### Contrast
Observed:
- low threshold 30
- high threshold 80
- max analysis side 1280

Confirmed:
- low 669
- normal 4635
- high 15
- unreadable 0

### Resolution
Confirmed:
- tiny 0
- small 0
- medium 1
- large 5318

### Aspect ratio
Confirmed:
- square 1
- landscape 5303
- portrait 15

### Class balance
Important bug fixed.

Original implementation only searched train/val/test, causing:
`{"classes": {}, "total": 0}`

Real dataset has classes at root. Module was updated to support class-folder layout.

Confirmed fixed output:
- layout: class_folders
- total: 5319
- natural_rock 1438 (27.04%)
- stone_artefact 3881 (72.96%)

Tests were added.

### Quality score
Confirmed:
- score 80/100
- MD5 duplicate groups 0
- visual duplicate groups 0
- blurry images 1782
- dark/bright images 2525
- tiny images 0

A `quality_scan.py` module was introduced during performance optimization.

### Statistics
Reads expected report JSONs:
- duplicates.json
- blur.json
- brightness.json
- contrast.json
- resolution.json
- aspect_ratio.json
- class_balance.json

Running statistics against an empty new report directory correctly showed missing markers. Running pipeline first populated them.

### Report
Generates human-readable `report.html`.

## Pipeline and performance

Original pipeline sequence observed:
verify -> analyze -> duplicates -> blur -> brightness -> contrast -> resolution -> aspect -> classes -> quality -> statistics -> report.

Original full real run:
`851.77 s`

Notable slow stages:
- blur ~225.41 s
- brightness ~384.64 s
- contrast ~237.59 s

Images are often 4K-6K, so repeated full decoding was expensive.

After optimization, benchmark:
`309.50 s`

Approximate elapsed-time reduction: 63.7%.

A later pipeline showed 10 steps, indicating some operations were consolidated. Inspect current `pipeline.py`; do not assume the old 12-step structure is still exact.

Do not reintroduce multiple full-resolution image passes without reason.

## DINOv2, FAISS and semantic analysis

Known modules include:
- embeddings/build.py
- embeddings/index.py
- embeddings/search.py
- embeddings/audit.py
- embeddings/outliers.py
- embeddings/outlier_review.py

Conceptual flow:
images -> DINOv2 embeddings -> FAISS index -> similarity search / semantic audit / outlier detection.

Legacy/root builder full-mode logic included cache detection for:
- `outputs/embeddings/dinov2_embeddings.npy`
- `outputs/embeddings/dinov2_paths.json`

If cached, it skips rebuilding embeddings. Otherwise it builds with batch size 16.

Then:
- build_index()
- semantic_audit(k=10)
- detect_outliers(k=10, percentile=2.0)
- build_outlier_contact_sheet(limit=30)

Outlier logic is known to:
- load embeddings and paths;
- normalize vectors;
- group by class;
- use `faiss.IndexFlatIP`;
- query nearest neighbors;
- exclude self;
- compute mean similarity;
- threshold by configurable percentile;
- write outlier candidates.

Outlier review generates a contact sheet for human review.

## Splitter — critical reusable component

Confirmed file:
`src/archaeosort_dataset_builder/splitter/dataset_splitter.py`

Do NOT rewrite from scratch.

Confirmed functions:
- parse_args
- validate_ratios
- discover_images_by_class
- split_class_images
- prepare_output_directory
- transfer_file
- build_split_dataset
- save_report
- main

Behavior:

`validate_ratios`:
- requires train + val + test == 1.0
- rejects negative ratios.

`discover_images_by_class`:
- recursive image discovery;
- parent directory name becomes class;
- FileNotFoundError if source missing;
- ValueError if no classified images.

`split_class_images`:
- deterministic `random.Random(seed).shuffle`;
- train count = int(total * train_ratio);
- val count = int(total * val_ratio);
- remainder test.

`prepare_output_directory`:
- refuses non-empty output unless force;
- with force removes/recreates it.

`transfer_file`:
- explicit copy mode uses shutil.copy2;
- otherwise tries hardlink;
- falls back to copy on OSError.

`build_split_dataset`:
- classes sorted;
- uses seed + class_index;
- creates train/val/test/class structure;
- names files `{index:06d}_{original_name}`;
- records per-split/per-class counts and total.

`save_report`:
- writes `split_report.json`.

Known cleanup:
some Spanish text had mojibake such as `imÃ¡genes`.
Module still has legacy argparse main; public interface should remain Typer.

## Exporter

Confirmed file:
`src/archaeosort_dataset_builder/exporter/dataset_exporter.py`

Current behavior is basic:
- argparse input-dir/output-dir;
- checks input exists;
- if output exists, deletes it unconditionally;
- shutil.copytree;
- prints success.

This is NOT safe enough for final public behavior.

Future integration:
- safe by default;
- respect force;
- callable API;
- avoid duplicate transfer logic;
- integrate with provenance/versioning if appropriate.

## Versioning

Confirmed file:
`src/archaeosort_dataset_builder/versioning/dataset_version.py`

Confirmed functions:
- `sha256_directory(directory)`
- `create_manifest(dataset_dir)`
- `save_manifest(dataset_dir)`

Manifest currently includes:
- timestamp-derived version `YYYY.MM.DD.HHMMSS`
- created UTC ISO timestamp
- dataset directory name
- file count
- size_bytes
- sha256

Current SHA implementation:
- sorts recursive files;
- hashes relative path;
- hashes complete file bytes.

Known issues to resolve:
1. `read_bytes()` loads a complete file into memory; chunked hashing is safer.
2. If `manifest.json` already exists, it participates in later hashes.
3. Timestamp-bearing generated metadata can make a fingerprint non-stable if included.
4. Define and test a deterministic hashing contract before calling versioning production-ready.

## Tests

Latest confirmed:
`31 passed in 2.25s`

Earlier checkpoints:
- 26 passed
- 30 passed
- 31 passed

Known test additions/refactors:
- class balance
- quality scan
- pipeline
- outliers
- existing validator/exporter/splitter/statistics affected by refactors.

Pipeline test incident:
a test monkeypatched module attributes like `pipeline.blur`; refactoring removed that attribute and produced AttributeError. Tests/code were adjusted and suite returned green.

Do not reduce coverage merely to make tests green.

### Required tests for build

Use pytest tmp_path and tiny generated images, not the real 5319-image dataset.

Test at minimum:
1. ratios sum to 1;
2. negative ratio rejected;
3. missing source rejected;
4. empty/unclassified source rejected;
5. non-empty output without force rejected;
6. force rebuild works;
7. same seed deterministic;
8. train/val/test generated;
9. class directories preserved;
10. total output images == source;
11. split_report exists and correct;
12. provenance exists and records seed/ratios/classes;
13. manifest exists;
14. input == output rejected;
15. copy mode;
16. hardlink fallback safe;
17. CLI build invokes engine.

## Ruff / quality gates

Observed Ruff config:
- line-length 100
- target-version py312

Routine:
`uv run --no-sync ruff check .`
`uv run --no-sync pytest -q`

Auto-fix only when appropriate:
`uv run --no-sync ruff check . --fix`

For syntax-sensitive edits:
`uv run --no-sync python -m py_compile <file>`

For packaging/CLI:
`uv sync`
`uv run archaeosort --help`

Before commit:
`git diff`
`git status`

## Dependencies observed

Main dependencies include:
pyyaml, requests, tqdm, pillow, pandas, numpy, opencv-python, imagehash, scikit-learn, kaggle, typer, pydantic, rich, torch, torchvision, timm, faiss-cpu, transformers, accelerate.

Dev:
mypy, pytest, pytest-cov, ruff.

Pytest:
- pythonpath = ["src"]
- testpaths = ["tests"]

## Git history milestones

Observed commits:
- `d69c82b feat: add dataset quality and semantic analysis pipeline`
- `5cc6efb test: expand test suite and add CI workflow`
- `9950423 docs: add bilingual project documentation`
- `4aff82c perf: optimize image quality analysis pipeline`
- `454d68d chore: ignore generated benchmark and test reports`
- `4401f7a feat: add installable Typer CLI`

Generated `reports_benchmark` and `reports_test` were accidentally included in the performance commit. `.gitignore` was subsequently updated. Do not add new generated report/output trees.

Git sync state changed during work, so inspect current status instead of trusting historical state.

## Documentation

Known:
- README.md
- README_ES.md
- GitHub CI workflow

README should ultimately document:
- uv installation;
- CLI;
- supported layouts;
- commands;
- reports;
- semantic workflow;
- build;
- output layout;
- provenance;
- manifest/fingerprint semantics;
- limitations;
- performance;
- architecture;
- real example results;
- development/testing.

## Known technical debt

- legacy argparse mains coexist with Typer;
- some mojibake strings;
- exporter destructive by default;
- version hashing needs deterministic contract/chunked reads;
- root builder.py is legacy and should eventually be compatibility wrapper or deprecated deliberately;
- heavy imports should remain out of basic CLI startup;
- semantic artifacts are large and must stay out of Git;
- JSON report schemas should evolve deliberately;
- reproducibility must mean more than simply having a seed.

## Immediate milestone: `archaeosort build`

Target command:

```powershell
uv run archaeosort `
  --dataset "C:\Users\farqu\Desktop\proyectos\archaeosort-datasets\datasets\v2\stone_figshare" `
  build `
  --output ".\datasets\stone_figshare\v1" `
  --train 0.70 `
  --val 0.15 `
  --test 0.15 `
  --seed 42
```

Desired optional flags:
- --copy
- --force

Target output:

```text
datasets/stone_figshare/v1/
├── train/
│   ├── natural_rock/
│   └── stone_artefact/
├── val/
│   ├── natural_rock/
│   └── stone_artefact/
├── test/
│   ├── natural_rock/
│   └── stone_artefact/
├── split_report.json
├── provenance.json
└── manifest.json
```

Build engine should:
1. resolve paths;
2. validate ratios;
3. reject input == output;
4. prepare output safely;
5. discover classes;
6. reuse existing deterministic splitter;
7. save split report;
8. save provenance;
9. generate deterministic manifest/fingerprint;
10. return structured result.

Provenance should include at least:
- created UTC timestamp
- source dataset
- output dataset
- seed
- mode copy vs hardlink_or_copy
- train/val/test ratios
- classes
- source image count

Potential future provenance fields:
- package version
- Git commit
- source manifest hash
- schema/builder version
- transformation/cleaning policy

## Critical uncertainty at handoff

A `builder/dataset_builder.py` implementation was proposed in the conversation, but there is NO confirmation that the user executed that block.

Therefore the next agent MUST verify:

```powershell
Test-Path .\src\archaeosort_dataset_builder\builder\dataset_builder.py
```

Also verify whether CLI already contains build:

```powershell
Select-String .\src\archaeosort_dataset_builder\cli.py `
  -Pattern 'command\("build"\)|def build_'
```

Do not overwrite an existing implementation without inspecting it.

## Recovery block for the next session

```powershell
cd C:\Users\farqu\Desktop\proyectos\archaeosort-dataset-builder

Write-Host "`n===== GIT ====="
git status --short
git log -5 --oneline

Write-Host "`n===== BUILD ENGINE ====="
Test-Path .\src\archaeosort_dataset_builder\builder\dataset_builder.py

if (Test-Path .\src\archaeosort_dataset_builder\builder\dataset_builder.py) {
    Get-Content .\src\archaeosort_dataset_builder\builder\dataset_builder.py
}

Write-Host "`n===== BUILD CLI ====="
Select-String .\src\archaeosort_dataset_builder\cli.py `
    -Pattern 'command\("build"\)|def build_'

Write-Host "`n===== CLI ====="
uv run archaeosort --help

Write-Host "`n===== RUFF ====="
uv run --no-sync ruff check .

Write-Host "`n===== TESTS ====="
uv run --no-sync pytest -q
```

Interpretation:
- engine absent -> create engine;
- engine present but CLI absent -> inspect/test engine, then wire CLI;
- engine + CLI present -> test tiny fixture before real dataset.

## Recommended implementation order

Phase 1: recover and inspect actual state.

Phase 2: implement/refine build engine.

Phase 3: implement build unit tests.

Phase 4: connect Typer build command.

Phase 5: add CLI tests.

Phase 6: run tiny synthetic dataset smoke test.

Phase 7: run real stone_figshare build into a separate output directory.

Phase 8: analyze the generated dataset with ArchaeoSort itself.

Phase 9: update README EN/ES.

Phase 10: Ruff + Pytest + Git diff/status + focused commit.

Suggested commit after successful milestone:
`feat: add reproducible versioned dataset build command`

## Definition of done for build

Not done until:
- build appears in CLI help;
- no hardcoded source;
- ratios validated;
- source/output safety;
- deterministic same-seed split;
- class distribution preserved;
- source not modified;
- hardlink/copy documented;
- force required for destructive overwrite;
- split_report.json;
- provenance.json;
- manifest.json;
- fingerprint semantics tested/documented;
- Ruff passes;
- tests pass;
- CI passes;
- real 5319-image smoke test succeeds;
- README EN/ES updated;
- generated dataset not committed.

## Longer-term ArchaeoSort-AI roadmap

After Dataset Builder:
1. baseline classifier;
2. transfer-learning comparison;
3. metrics/confusion matrix;
4. model export;
5. FastAPI inference;
6. Next.js/TypeScript frontend;
7. upload/inference UI;
8. similarity explorer;
9. active learning/review;
10. PostgreSQL prediction/review history;
11. optional Flutter mobile app;
12. Docker;
13. CI/CD;
14. model/data cards;
15. portfolio demo and LinkedIn material.

Previously preferred broader stack:
- FastAPI
- PostgreSQL
- Next.js + TypeScript
- Tailwind
- shadcn/ui
- Framer Motion
- Recharts
- vector database only if justified
- Flutter later

These are future stages, not requirements for the current Dataset Builder milestone.

## User workflow requirements

The user wants:
- terminal-first work;
- PowerShell;
- exact step-by-step commands;
- professional architecture;
- reuse existing code;
- explain what each step proves;
- preserve errors and fixes for project documentation;
- GitHub-ready result;
- portfolio/business relevance;
- AI-agent acceleration without losing technical coherence.

When working interactively, prefer:
1. one coherent change;
2. exact command(s);
3. expected result;
4. explanation;
5. wait for terminal output before assuming risky structural changes succeeded.

## Historical errors to remember

`--dataset` unrecognized:
CLI did not expose runtime dataset; fixed with global options/settings.

`NoneType.rglob`:
dataset was not propagated after making setting optional; fix by explicit path or require_dataset.

Class balance empty:
code assumed train/val/test; fixed to support root class folders.

`ModuleNotFoundError: archaeosort_dataset_builder`:
plain Python/environment mismatch; use project package via uv.

`archaeosort` not found:
entry point/package not installed; fixed with project scripts/build system + uv sync.

OpenBLAS memory allocation:
avoid eager heavy imports in basic CLI.

Ruff B008:
use Typer Annotated patterns.

SyntaxError after mechanical edit:
compile immediately after structural transformations.

Generated reports committed:
keep generated report/benchmark/output trees ignored.

## Canonical confirmed-state summary

CONFIRMED:
- installable Python project structure;
- generalized runtime settings;
- Typer CLI works;
- `uv run archaeosort --help` works;
- 19 analysis/semantic commands exposed;
- 5319-image real dataset analyzed;
- root class-folder support fixed;
- pipeline performance improved from 851.77s to 309.50s;
- DINOv2/FAISS/semantic/outlier components exist;
- JSON + HTML reporting exists;
- bilingual README documentation exists;
- CI exists;
- latest confirmed tests: 31 passed;
- splitter exists and is reusable;
- exporter exists but is basic;
- versioning exists but needs hashing refinement.

NOT CONFIRMED:
- whether proposed `builder/dataset_builder.py` exists;
- build-specific tests;
- `archaeosort build`;
- successful versioned train/val/test output through new CLI;
- exact current remote Git synchronization after all packaging edits.

FIRST ACTION: inspect reality, then continue.

## Recommended first prompt for an AI coding agent

Continue development of the existing `archaeosort-dataset-builder` repository using the attached handoff as context. First inspect the actual repository and compare it against the handoff. Do not modify code yet. Check Git status/log, current CLI help, Ruff, Pytest, whether `src/archaeosort_dataset_builder/builder/dataset_builder.py` exists, and whether Typer already exposes `build`. Report the exact current state and any differences. Then propose the smallest safe next implementation step toward a tested, reproducible `archaeosort build` command. Do not assume previously proposed code was executed.
