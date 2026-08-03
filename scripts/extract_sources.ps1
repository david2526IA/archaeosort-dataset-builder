$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$downloadsPath = Join-Path $projectRoot "data\downloads"
$stagingPath = Join-Path $projectRoot "data\staging"
$metadataPath = Join-Path $projectRoot "data\metadata"

$env:PYTHONPATH = Join-Path $projectRoot "src"

& $pythonPath -m archaeosort_dataset_builder.processors.extract_sources `
    --downloads-dir $downloadsPath `
    --staging-dir $stagingPath `
    --metadata-dir $metadataPath