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
    If you get a "running scripts is disabled" error, run this first:
      Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

    Conda must be initialized for PowerShell. If 'conda activate' fails, run:
      conda init powershell
    then restart your terminal.

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

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: conda is not available. Install Miniconda or Anaconda first:"
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

# ── Step 1: Create or reuse conda environment ────────────────────────────────

Write-Host ""
$envExists = conda env list | Select-String -Pattern "^\s*$EnvName\s" -Quiet
if (-not $envExists) {
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
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate conda environment '$EnvName'."
    Write-Host "       Make sure you have run 'conda init powershell' first."
    exit 1
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
