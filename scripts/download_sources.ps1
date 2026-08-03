$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================="
Write-Host " ArchaeoSort - Download Kaggle Sources"
Write-Host "========================================="
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\sources.yaml"
$downloadsPath = Join-Path $projectRoot "data\downloads"
$metadataPath = Join-Path $projectRoot "data\metadata"

$env:PYTHONPATH = Join-Path $projectRoot "src"

& $pythonPath -m archaeosort_dataset_builder.downloaders.kaggle_downloader `
    --config $configPath `
    --output-dir $downloadsPath `
    --metadata-dir $metadataPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] La descarga ha fallado."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[OK] Descarga completada."
