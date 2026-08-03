$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================="
Write-Host " ArchaeoSort - Validate Sources Manifest"
Write-Host "========================================="
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\sources.yaml"
$outputPath = Join-Path $projectRoot "outputs\manifest_validation.json"

if (!(Test-Path $pythonPath)) {
    Write-Host "[ERROR] No se encontro el entorno virtual:"
    Write-Host $pythonPath
    exit 1
}

if (!(Test-Path $configPath)) {
    Write-Host "[ERROR] No se encontro el manifiesto:"
    Write-Host $configPath
    exit 1
}

$env:PYTHONPATH = Join-Path $projectRoot "src"

& $pythonPath -m archaeosort_dataset_builder.manifest `
    --config $configPath `
    --output $outputPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] El manifiesto contiene errores."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[OK] Manifiesto validado correctamente."
