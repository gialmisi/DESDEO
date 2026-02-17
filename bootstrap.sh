#!/usr/bin/env bash
# Quick-start script for DESDEO.
#
# Usage:
#   curl -sSf <repo-url>/bootstrap.sh | bash
#   — or —
#   ./bootstrap.sh
#
# Installs uv (if missing), syncs Python dependencies, then launches
# the interactive setup wizard.

set -euo pipefail

echo ""
echo "========================================"
echo "  DESDEO Bootstrap"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Install uv (Python package manager) if not already installed"
echo "  2. Sync Python dependencies via 'uv sync'"
echo "  3. Launch the interactive DESDEO setup wizard"
echo ""

read -rp "Continue? [Y/n] " answer
case "${answer:-y}" in
    [yY]|[yY][eE][sS]) ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac

# ── Step 1: Install uv if not available ─────────────────────────────────────

echo ""
if ! command -v uv &>/dev/null; then
    echo "[1/3] uv not found — installing from https://astral.sh/uv/ ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    echo "       Installed: $(uv --version)"
else
    echo "[1/3] uv already installed: $(uv --version)"
fi

# ── Step 2: Sync Python dependencies ───────────────────────────────────────

echo ""
echo "[2/3] Syncing Python dependencies (this may take a moment on first run)..."
uv sync

# ── Step 3: Launch the setup wizard ─────────────────────────────────────────

echo ""
echo "[3/3] Launching DESDEO setup wizard..."
echo ""
exec uv run desdeo-setup
