#!/bin/bash

# ANSI color codes
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to prepend colored (Backend) to each line of output
prepend_backend() {
    while IFS= read -r line; do
        echo -e "${BLUE}(Backend)${NC} $line"
    done
}

# Function to prepend colored (Frontend) to each line of output
prepend_frontend() {
    while IFS= read -r line; do
        echo -e "${YELLOW}(Frontend)${NC} $line"
    done
}

# Function to kill background processes
cleanup() {
    echo "Shutting down..."
    kill -TERM $backend_pid $frontend_pid
    wait $backend_pid $frontend_pid
    exit 0
}

# Read paths from .desdeo/config.toml if present
DESDEO_CONFIG=".desdeo/config.toml"
if [ -z "$NVM_DIR" ] && [ -f "$DESDEO_CONFIG" ]; then
    NVM_DIR=$(grep '^nvm_dir' "$DESDEO_CONFIG" | sed 's/^nvm_dir *= *"\(.*\)"/\1/')
fi
if [ -f "$DESDEO_CONFIG" ]; then
    _solver_dir=$(grep '^solvers' "$DESDEO_CONFIG" | sed 's/^solvers *= *"\(.*\)"/\1/')
    [ -n "$_solver_dir" ] && [ -d "$_solver_dir" ] && export PATH="$_solver_dir:$PATH"
fi

# Source nvm if available (so npm/node are on PATH)
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Run Uvicorn from the ./desdeo/api directory and prepend colored (Backend) to its output
# Use "uv run" if uv is available, otherwise call uvicorn directly
if command -v uv &>/dev/null; then
    UVICORN="uv run uvicorn"
else
    UVICORN="uvicorn"
fi

(cd ./desdeo/api && $UVICORN app:app --reload --log-level debug --host 127.0.0.1 --port 8000) | prepend_backend &
backend_pid=$!

# Run npm command from the ./webui directory and prepend colored (Frontend) to its output
(cd ./webui && npm run dev -- --open) | prepend_frontend &
frontend_pid=$!

wait $backend_pid $frontend_pid
