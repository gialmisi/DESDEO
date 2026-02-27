#Requires -Version 5.1
<#
.SYNOPSIS
    Conda bootstrap script for DESDEO.

.DESCRIPTION
    Creates a conda environment, installs uv via conda-forge, installs the
    project in editable mode with all dev dependencies, then launches the
    interactive setup wizard.

.PARAMETER EnvName
    Name for the conda environment (default: desdeo).

.NOTES
    Prerequisite: conda must be available on PATH.

    If you get a "running scripts is disabled" error, run this first:
      Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

.EXAMPLE
    .\bootstrap-conda.ps1
    .\bootstrap-conda.ps1 -EnvName myenv
#>

param(
    [string]$EnvName = "desdeo"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "  DESDEO Conda Bootstrap"
Write-Host "========================================"
Write-Host ""
Write-Host "This script will:"
Write-Host "  1. Create (or reuse) conda environment '$EnvName' with Python 3.12"
Write-Host "  2. Install uv via conda-forge"
Write-Host "  3. Install DESDEO (editable) into the conda environment"
Write-Host "  4. Install all development dependencies (--group all-dev)"
Write-Host "  5. Launch the interactive DESDEO setup wizard"
Write-Host ""

$answer = Read-Host "Continue? [Y/n]"
if ($answer -match '^[nN]') {
    Write-Host "Aborted."
    exit 0
}

# ── Pre-check: conda available? ──────────────────────────────────────────────

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Host ""
    Write-Host "ERROR: conda is not available on PATH."
    Write-Host "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

# ── Initialize conda shell hook if needed ────────────────────────────────────
# If the user ran 'conda init powershell', conda is already a PowerShell
# function and activate works out of the box.  Otherwise conda resolves to
# conda.bat/.exe on PATH and we must import Conda.psm1 to make activate work.

if ($condaCmd.CommandType -ne 'Function') {
    $condaExePath = $condaCmd.Source
    $condaRoot = (Split-Path (Split-Path $condaExePath))
    $condaModule = Join-Path $condaRoot "shell" "condabin" "Conda.psm1"
    if (-not (Test-Path $condaModule)) {
        # Some layouts nest condabin one level deeper
        $condaRoot = (Split-Path $condaRoot)
        $condaModule = Join-Path $condaRoot "shell" "condabin" "Conda.psm1"
    }
    if (Test-Path $condaModule) {
        Import-Module $condaModule
    } else {
        Write-Host ""
        Write-Host "ERROR: Could not find Conda.psm1 to enable 'conda activate'."
        Write-Host "       Run 'conda init powershell', restart PowerShell, and try again."
        exit 1
    }
}

# ── Step 1: Create or reuse conda environment ────────────────────────────────

Write-Host ""
$envMatch = conda env list | Select-String -Pattern "^$([regex]::Escape($EnvName))\s"
if (-not $envMatch) {
    Write-Host "[1/5] Creating conda environment '$EnvName' with Python 3.12..."
    conda create -y -n $EnvName python=3.12
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create conda environment."
        exit 1
    }
} else {
    Write-Host "[1/5] Conda environment '$EnvName' already exists - reusing it."
}

Write-Host "       Activating '$EnvName'..."
conda activate $EnvName
if ($env:CONDA_DEFAULT_ENV -ne $EnvName) {
    Write-Host "ERROR: Failed to activate conda environment '$EnvName'."
    exit 1
}

# ── Set uv cache inside the conda env ─────────────────────────────────────────
# On restricted Windows machines the default uv cache (AppData\Local\uv\cache)
# may lack write permissions or trigger admin prompts. Placing the cache inside
# the conda env avoids this.

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $env:CONDA_PREFIX "uv_cache"
    Write-Host "       UV_CACHE_DIR set to $env:UV_CACHE_DIR"
}

# ── Step 2: Install uv via conda-forge ────────────────────────────────────────

Write-Host ""
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[2/5] Installing uv via conda-forge..."
    conda install -y conda-forge::uv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: uv installation failed."
        exit 1
    }
    Write-Host "       Installed: $(uv --version)"
} else {
    Write-Host "[2/5] uv already installed: $(uv --version)"
}

# ── Step 3: Install DESDEO in editable mode ───────────────────────────────────

Write-Host ""
Write-Host "[3/5] Installing DESDEO in editable mode..."
uv pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install DESDEO."
    exit 1
}

# ── Step 4: Install all development dependencies ──────────────────────────────

Write-Host ""
Write-Host "[4/5] Installing all development dependencies (this may take a moment)..."
uv pip install --group all-dev
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install development dependencies."
    exit 1
}

# ── Step 5: Launch the setup wizard ───────────────────────────────────────────

Write-Host ""
Write-Host "[5/5] Launching DESDEO setup wizard..."
Write-Host ""
desdeo-setup

Write-Host ""
Write-Host "========================================"
Write-Host "  Setup complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "To start working, run:"
Write-Host "  conda activate $EnvName"
Write-Host ""
