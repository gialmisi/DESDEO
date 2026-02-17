#!/usr/bin/env bash
# ============================================================================
# e2e_cli_test.sh — End-to-end test for the desdeo-setup CLI
#
# Runs inside Docker (see Dockerfile.cli-test).  Simulates a brand-new user
# on a fresh machine: downloads real solvers, installs Node via the CLI's
# built-in nvm support, runs real npm install, creates a real SQLite database,
# then verifies the CLI reports everything green.
# ============================================================================

set -o pipefail

# ── Counters & helpers ──────────────────────────────────────────────────────

PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo "  ✓ PASS: $1"
    ((PASS_COUNT++)) || true
}

fail() {
    echo "  ✗ FAIL: $1"
    ((FAIL_COUNT++)) || true
}

assert_contains() {
    local haystack="$1" needle="$2" label="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        pass "$label"
    else
        fail "$label — expected to find '$needle'"
    fi
}

assert_file_exists() {
    local path="$1" label="$2"
    if [ -e "$path" ]; then
        pass "$label"
    else
        fail "$label — not found: $path"
    fi
}

assert_on_path() {
    local cmd="$1" label="$2"
    if command -v "$cmd" &>/dev/null; then
        pass "$label"
    else
        fail "$label — '$cmd' not on PATH"
    fi
}

assert_not_on_path() {
    local cmd="$1" label="$2"
    if ! command -v "$cmd" &>/dev/null; then
        pass "$label"
    else
        fail "$label — '$cmd' unexpectedly found on PATH"
    fi
}

assert_file_not_exists() {
    local path="$1" label="$2"
    if [ ! -e "$path" ]; then
        pass "$label"
    else
        fail "$label — unexpectedly exists: $path"
    fi
}

phase_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Phase $1: $2"
    echo "════════════════════════════════════════════════════════════════"
}

# ── Phase 1: Verify Fresh State ────────────────────────────────────────────

phase_header 1 "Verify Fresh State"

output=$(uv run desdeo-setup check 2>&1) || true
echo "$output"

# Tools that the CLI will set up must NOT be present yet
assert_not_on_path bonmin "bonmin not on PATH (fresh)"
assert_not_on_path ipopt  "ipopt not on PATH (fresh)"
assert_not_on_path cbc    "cbc not on PATH (fresh)"
assert_not_on_path node   "node not on PATH (fresh)"
assert_not_on_path npm    "npm not on PATH (fresh)"
assert_not_on_path nvm    "nvm not on PATH (fresh)"

# Artifacts must not exist
assert_file_not_exists /app/desdeo/api/test.db    "no database file (fresh)"
assert_file_not_exists /app/webui/node_modules    "no node_modules (fresh)"
assert_file_not_exists /app/webui/.env            "no webui/.env (fresh)"

# The check command should report issues
assert_contains "$output" "Issues found" "check detects issues in fresh state"

# ── Phase 1.5: Create Install Config ──────────────────────────────────────

phase_header 1.5 "Install Config"

# Create .desdeo/config.toml with default paths so subsequent phases
# read from config instead of re-prompting for install locations.
uv run python -c "
from desdeo.cli.config import InstallMode, resolve_paths_for_mode, save_config
paths = resolve_paths_for_mode(InstallMode.DEFAULT)
save_config(mode=InstallMode.DEFAULT, solvers=paths['solvers'], nvm_dir=paths['nvm_dir'])
print('Config created')
" 2>&1

assert_file_exists /app/.desdeo/config.toml "config file created"

# ── Phase 2: Solver Setup (real download) ──────────────────────────────────

phase_header 2 "Solver Setup"

# Prompt sequence (solvers.py):
#   1  = download from DESDEO GitHub releases
#   (no install location prompt — config exists)
#   y  = add to PATH in ~/.bashrc
#
# Gurobi: gurobipy is not installed in this image, so _setup_gurobi()
# prints info and returns — NO prompt is shown.
printf '%s\n' 1 y | uv run desdeo-setup solvers 2>&1

# Sourcing .bashrc may fail (non-interactive guards), so set PATH directly.
SOLVER_DIR="$HOME/.local/share/desdeo/solvers"
export PATH="$SOLVER_DIR:$PATH"

assert_on_path bonmin "bonmin on PATH after setup"
assert_on_path ipopt  "ipopt on PATH after setup"
assert_on_path cbc    "cbc on PATH after setup"

# Verify bashrc was updated
if grep -q "DESDEO solvers" ~/.bashrc 2>/dev/null; then
    pass "~/.bashrc contains solver PATH entry"
else
    fail "~/.bashrc missing solver PATH entry"
fi

# ── Phase 3: WebUI Setup (installs Node via nvm + npm install) ─────────────

phase_header 3 "WebUI Setup"

# Prompt sequence (webui.py):
#   1        = install Node.js 24 via nvm (the new menu)
#   (empty)  = accept default API_BASE_URL  (http://localhost:8000)
#   (empty)  = accept default VITE_API_URL  (/api)
printf '%s\n' 1 '' '' | uv run desdeo-setup webui 2>&1

# Source nvm so node/npm are on PATH for subsequent phases
export NVM_DIR="$HOME/.nvm"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

assert_on_path node "Node.js installed by CLI"
assert_on_path npm  "npm available after CLI install"

assert_file_exists /app/webui/node_modules "webui/node_modules created"
assert_file_exists /app/webui/.env         "webui/.env created"

env_content=$(cat /app/webui/.env)
assert_contains "$env_content" "API_BASE_URL" ".env has API_BASE_URL"
assert_contains "$env_content" "VITE_API_URL" ".env has VITE_API_URL"

# ── Phase 4: Database Setup (real SQLite creation) ─────────────────────────

phase_header 4 "Database Setup"

# Prompt sequence (database.py):
#   1        = SQLite mode
#   (empty)  = accept default DB URL
#   ---- _handle_existing_db: DB doesn't exist yet → creates it, no prompt ----
#   analyst  = analyst username
#   analyst  = analyst password  (hide_input; getpass falls back to stdin pipe)
#   test     = analyst group
#   3        = skip DM user creation
#   1        = default problems  (dtlz2, simple_knapsack, river_pollution_problem)
#   10       = dtlz2 n_variables
#   3        = dtlz2 n_objectives
printf '%s\n' 1 '' analyst analyst test 3 1 10 3 | uv run desdeo-setup db 2>&1

assert_file_exists /app/desdeo/api/test.db "database file created"

# Query the database to verify
db_check=$(uv run python -c "
from sqlmodel import Session, create_engine, select
import desdeo.api.models
from desdeo.api.models import User

engine = create_engine(
    'sqlite:////app/desdeo/api/test.db',
    connect_args={'check_same_thread': False},
)
with Session(engine) as s:
    users = s.exec(select(User)).all()
    names = [u.username for u in users]
    print(f'users={names}')
    assert 'analyst' in names, 'analyst user not found'
print('DB OK')
" 2>&1) && pass "database contains analyst user" || fail "database verification"
echo "  $db_check"

# ── Phase 5: Full Re-check ─────────────────────────────────────────────────

phase_header 5 "Full Re-check"

output=$(uv run desdeo-setup check 2>&1) || true
echo "$output"

assert_contains "$output" "Everything looks good" "all required checks pass"

# ── Phase 6: Fullstack Smoke Test (optional) ───────────────────────────────

phase_header 6 "Fullstack Smoke Test"

if [ "${RUN_FULLSTACK:-0}" = "1" ]; then
    echo "  Starting backend..."
    uv run uvicorn desdeo.api.app:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    echo "  Starting frontend..."
    (
        cd /app/webui
        source "$NVM_DIR/nvm.sh"
        npx vite --host 0.0.0.0 --port 5173 2>&1
    ) &
    FRONTEND_PID=$!

    echo "  Waiting 15s for services..."
    sleep 15

    if curl -sf http://localhost:8000/docs > /dev/null 2>&1; then
        pass "backend responds on :8000"
    else
        fail "backend not responding on :8000"
    fi

    if curl -sf http://localhost:5173 > /dev/null 2>&1; then
        pass "frontend responds on :5173"
    else
        fail "frontend not responding on :5173"
    fi

    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
else
    echo "  Skipped (set RUN_FULLSTACK=1 to enable)"
fi

# ── Phase 7: Idempotent Re-run ─────────────────────────────────────────────

phase_header 7 "Idempotent Re-run"

# The wizard should detect everything is set up and exit with no prompts.
output=$(uv run desdeo-setup 2>&1) || true
echo "$output"

assert_contains "$output" "Everything is already set up" "wizard detects complete setup"

# ── Summary ────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════"
total=$((PASS_COUNT + FAIL_COUNT))
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "  ALL PASSED: $PASS_COUNT / $total assertions"
else
    echo "  FAILED: $FAIL_COUNT / $total assertions failed"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

[ "$FAIL_COUNT" -eq 0 ]
