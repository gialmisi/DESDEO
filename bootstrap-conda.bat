@echo off
REM Conda bootstrap script for DESDEO.
REM
REM Usage:
REM   bootstrap-conda.bat [env-name]
REM
REM Creates a conda environment, installs uv via conda-forge, installs the
REM project in editable mode with all dev dependencies, then launches the
REM interactive setup wizard.

setlocal enabledelayedexpansion

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=desdeo"

echo.
echo ========================================
echo   DESDEO Conda Bootstrap
echo ========================================
echo.
echo This script will:
echo   1. Create (or reuse) conda environment '%ENV_NAME%' with Python 3.12
echo   2. Install uv via conda-forge
echo   3. Install DESDEO (editable) into the conda environment
echo   4. Install all development dependencies (--group all-dev)
echo   5. Launch the interactive DESDEO setup wizard
echo.

set /p "answer=Continue? [Y/n] "
if /i "%answer%"=="n" (
    echo Aborted.
    exit /b 0
)
if /i "%answer%"=="no" (
    echo Aborted.
    exit /b 0
)

REM ── Pre-check: conda available? ────────────────────────────────────────────

where conda >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: conda is not available. Install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM ── Step 1: Create or reuse conda environment ──────────────────────────────

echo.
conda env list | findstr /c:"%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [1/5] Creating conda environment '%ENV_NAME%' with Python 3.12...
    call conda create -y -n %ENV_NAME% python=3.12
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment.
        exit /b 1
    )
) else (
    echo [1/5] Conda environment '%ENV_NAME%' already exists — reusing it.
)

echo        Activating '%ENV_NAME%'...
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment '%ENV_NAME%'.
    exit /b 1
)

REM ── Step 2: Install uv via conda-forge ──────────────────────────────────────

echo.
where uv >nul 2>&1
if errorlevel 1 (
    echo [2/5] Installing uv via conda-forge...
    call conda install -y conda-forge::uv
    if errorlevel 1 (
        echo ERROR: uv installation failed.
        exit /b 1
    )
    for /f "tokens=*" %%v in ('uv --version') do echo        Installed: %%v
) else (
    for /f "tokens=*" %%v in ('uv --version') do echo [2/5] uv already installed: %%v
)

REM ── Step 3: Install DESDEO in editable mode ─────────────────────────────────

echo.
echo [3/5] Installing DESDEO in editable mode...
uv pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install DESDEO.
    exit /b 1
)

REM ── Step 4: Install all development dependencies ────────────────────────────

echo.
echo [4/5] Installing all development dependencies (this may take a moment)...
uv pip install --group all-dev
if errorlevel 1 (
    echo ERROR: Failed to install development dependencies.
    exit /b 1
)

REM ── Step 5: Launch the setup wizard ─────────────────────────────────────────

echo.
echo [5/5] Launching DESDEO setup wizard...
echo.
desdeo-setup
