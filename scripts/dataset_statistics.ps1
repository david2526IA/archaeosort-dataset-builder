$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$datasetPath = Join-Path $projectRoot "data\staging\archaeomind_kaggle\ArchaeoMind_dataset"
$outputPath = Join-Path $projectRoot "outputs\statistics\dataset_summary.json"

if (!(Test-Path $pythonPath)) {
    Write-Host "[ERROR] No existe Python:"
    Write-Host $pythonPath
    exit 1
}

if (!(Test-Path $datasetPath)) {
    Write-Host "[ERROR] No existe el dataset:"
    Write-Host $datasetPath
    exit 1
}

$env:PYTHONPATH = Join-Path $projectRoot "src"

& $pythonPath -m archaeosort_dataset_builder.statistics.dataset_statistics `
    --dataset-dir $datasetPath `
    --output $outputPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Falló la generación de estadísticas."
    exit $LASTEXITCODE
}

Write-Host "[OK] Estadísticas generadas."
