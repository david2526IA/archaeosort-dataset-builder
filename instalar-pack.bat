@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "SRC=%ROOT%\md"
set "AUTO_CONFIRM=0"
set "DEST="

if /I "%~1"=="/Y" (
    set "AUTO_CONFIRM=1"
    shift
)
if /I "%~1"=="--yes" (
    set "AUTO_CONFIRM=1"
    shift
)
if /I "%~2"=="/Y" set "AUTO_CONFIRM=1"
if /I "%~2"=="--yes" set "AUTO_CONFIRM=1"

if not exist "%SRC%\README.md" (
    echo [ERROR] No se encontro la carpeta del pack:
    echo         %SRC%
    exit /b 1
)

if "%~1"=="" (
    set "DEST=%ROOT%"
    set "AUTO_CONFIRM=1"
) else (
    set "DEST=%~1"
)

if not defined DEST (
    echo [ERROR] No se indico carpeta destino.
    exit /b 1
)

for %%I in ("%DEST%") do set "DEST=%%~fI"

if /I "%DEST%"=="%SRC%" (
    echo [ERROR] El destino no puede ser la propia carpeta md.
    exit /b 1
)

echo.
echo Instalador del pack portable de instrucciones para IA
echo.
echo Origen:
echo   %SRC%
if "%~1"=="" (
    echo Destino automatico ^(directorio actual^):
) else (
    echo Destino indicado:
)
echo   %DEST%
echo.

echo.
echo Se copiara el contenido de:
echo   %SRC%
echo a:
echo   %DEST%
echo.
if "%AUTO_CONFIRM%"=="1" (
    echo Confirmacion automatica activada.
) else (
    choice /C SN /N /M "Continuar? [S/N]: "
    if errorlevel 2 (
        echo Cancelado.
        exit /b 0
    )
)

if not exist "%DEST%" (
    mkdir "%DEST%" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No se pudo crear la carpeta destino.
        exit /b 1
    )
)

robocopy "%SRC%" "%DEST%" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
    echo [ERROR] Fallo la copia. Codigo robocopy: %RC%
    exit /b %RC%
)

echo.
echo [OK] Pack instalado correctamente.
echo.
echo Siguiente paso recomendado:
echo - Abre el proyecto destino en tu herramienta de IA.
echo - Pide: Resume las reglas locales que estas siguiendo en este repo.
echo - Si la herramienta ignora archivos locales, pega PROMPT-MANUAL.md.

exit /b 0
