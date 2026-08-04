$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:PYTHONPATH = Join-Path $projectRoot "src"

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $python -m archaeosort_dataset_builder.splitter.dataset_splitter `
    --input-dir "data\staging\archaeomind_kaggle\ArchaeoMind_dataset" `
    --output-dir "outputs\dataset_v1" `
    --train-ratio 0.70 `
    --val-ratio 0.15 `
    --test-ratio 0.15 `
    --force
