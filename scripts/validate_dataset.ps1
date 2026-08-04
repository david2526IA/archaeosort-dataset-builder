$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:PYTHONPATH = Join-Path $projectRoot "src"

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $python -c @"

from pathlib import Path

from archaeosort_dataset_builder.validator.dataset_validator import validate_dataset
from archaeosort_dataset_builder.validator.report_generator import save_report

dataset = Path("data/processed/archaeomind_kaggle")

issues = validate_dataset(dataset)

output = Path("outputs/validation/validation_report.json")

save_report(issues, output)

print("=" * 60)
print("ARCHAEOSORT DATASET VALIDATOR")
print("=" * 60)
print(f"Issues found: {len(issues)}")
print(f"Report: {output.resolve()}")
print("=" * 60)

"@
