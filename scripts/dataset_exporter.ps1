$ErrorActionPreference="Stop"

$projectRoot=Split-Path -Parent $PSScriptRoot

Set-Location $projectRoot

$env:PYTHONPATH=Join-Path $projectRoot "src"

$python=Join-Path $projectRoot ".venv\Scripts\python.exe"

& $python -m archaeosort_dataset_builder.exporter.dataset_exporter `
    --input-dir "outputs\dataset_v1" `
    --output-dir "exports\imagefolder"
