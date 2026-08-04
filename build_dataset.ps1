$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (!(Test-Path $python)) {
    Write-Host ""
    Write-Host "[ERROR] No se encuentra Python:"
    Write-Host $python
    exit 1
}

$env:PYTHONPATH = Join-Path $projectRoot "src"

Write-Host ""
Write-Host "========================================"
Write-Host "ARCHAEOSORT DATASET BUILDER"
Write-Host "========================================"

Write-Host ""
Write-Host "[1/5] Dataset Statistics..."
& .\scripts\dataset_statistics.ps1

Write-Host ""
Write-Host "[2/5] Dataset Splitter..."
& .\scripts\dataset_splitter.ps1

Write-Host ""
Write-Host "[3/5] Dataset Exporter..."
& .\scripts\dataset_exporter.ps1

Write-Host ""
Write-Host "[4/5] Dataset Version..."
$env:PYTHONPATH="src"
& $python -m archaeosort_dataset_builder.versioning.dataset_version

Write-Host ""
Write-Host "[5/5] Git Status..."
git status

Write-Host ""
Write-Host "========================================"
Write-Host "PIPELINE FINISHED"
Write-Host "========================================"
