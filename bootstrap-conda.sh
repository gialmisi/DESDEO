#!/usr/bin/env bash
# Conda bootstrap script for DESDEO.
#
# Usage:
#   ./bootstrap-conda.sh [env-name]
#
# Creates a conda environment, installs uv via conda-forge, installs the
# project in editable mode with all dev dependencies, then launches the
# interactive setup wizard.

set -euo pipefail

ENV_NAME="${1:-desdeo}"

echo ""
echo "========================================"
echo "  DESDEO Conda Bootstrap"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Create (or reuse) conda environment '$ENV_NAME' with Python 3.12"
echo "  2. Install uv via conda-forge"
echo "  3. Install DESDEO (editable) into the conda environment"
echo "  4. Install all development dependencies (--group all-dev)"
echo "  5. Launch the interactive DESDEO setup wizard"
echo ""

read -rp "Continue? [Y/n] " answer
case "${answer:-y}" in
    [yY]|[yY][eE][sS]) ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac

# ── Pre-check: conda available? ──────────────────────────────────────────────

if ! command -v conda &>/dev/null; then
    echo ""
    echo "ERROR: conda is not available. Install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Enable conda activate in scripts
eval "$(conda shell.bash hook)"

# ── Step 1: Create or reuse conda environment ────────────────────────────────

echo ""
if conda env list | grep -qw "$ENV_NAME"; then
    echo "[1/5] Conda environment '$ENV_NAME' already exists — reusing it."
else
    echo "[1/5] Creating conda environment '$ENV_NAME' with Python 3.12..."
    conda create -y -n "$ENV_NAME" python=3.12
fi

echo "       Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"

# ── Step 2: Install uv via conda-forge ────────────────────────────────────────

echo ""
if command -v uv &>/dev/null; then
    echo "[2/5] uv already installed: $(uv --version)"
else
    echo "[2/5] Installing uv via conda-forge..."
    conda install -y conda-forge::uv
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed."
        exit 1
    fi
    echo "       Installed: $(uv --version)"
fi

# ── Step 3: Install DESDEO in editable mode ───────────────────────────────────

echo ""
echo "[3/5] Installing DESDEO in editable mode..."
uv pip install -e .

# ── Step 4: Install all development dependencies ─────────────────────────────

echo ""
echo "[4/5] Installing all development dependencies (this may take a moment)..."
uv pip install --group all-dev

# ── Step 5: Launch the setup wizard ───────────────────────────────────────────

echo ""
echo "[5/5] Launching DESDEO setup wizard..."
echo ""
exec desdeo-setup
