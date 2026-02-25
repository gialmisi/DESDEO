#!/usr/bin/env bash
# ============================================================================
# e2e_wizard_test.sh — End-to-end test for the unified desdeo-setup wizard
#
# Runs inside Docker (see Dockerfile.cli-test).  Unlike e2e_cli_test.sh which
# invokes each subcommand individually, this test runs a single
# `desdeo-setup` invocation with all prompts piped in sequence — exactly
# how a real user would experience it.
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

# Artifacts must not exist
assert_file_not_exists /app/desdeo/api/test.db    "no database file (fresh)"
assert_file_not_exists /app/webui/node_modules    "no node_modules (fresh)"
assert_file_not_exists /app/webui/.env            "no webui/.env (fresh)"
assert_file_not_exists /app/.desdeo/config.toml   "no config.toml (fresh)"

# The check command should report issues
assert_contains "$output" "Issues found" "check detects issues in fresh state"

# ── Phase 2: Run Full Wizard ───────────────────────────────────────────────

phase_header 2 "Run Full Wizard"

# Complete prompt sequence for `desdeo-setup` (no subcommand):
#
#   P1:  1        Install location: default           (wizard.py _configure_install_paths)
#   P2:  y        Apply to all components              (wizard.py _configure_install_paths)
#   P3:  y        Proceed with solver setup            (wizard.py setup)
#   P4:  1        Download from GitHub                 (solvers.py solvers)
#   P5:  y        Add to PATH in .bashrc              (solvers.py _add_to_path)
#   P6:  y        Proceed with database setup          (wizard.py setup)
#   P7:  1        SQLite mode                          (database.py db)
#   P8:  (empty)  Accept default DB URL                (database.py db)
#   P9:  analyst  Analyst username                     (database.py _setup_users)
#   P10: analyst  Analyst password (hide_input)        (database.py _setup_users)
#   P11: test     Analyst group                        (database.py _setup_users)
#   P12: 3        Skip DM user creation                (database.py _setup_users)
#   P13: 1        Default problems                     (database.py _setup_problems)
#   P14: 10       dtlz2 n_variables                    (database.py _instantiate_problem)
#   P15: 3        dtlz2 n_objectives                   (database.py _instantiate_problem)
#   P16: y        Proceed with WebUI setup             (wizard.py setup)
#   P17: 1        Install Node via nvm                 (webui.py _check_node_version)
#   P18: (empty)  Accept default API_BASE_URL          (webui.py _setup_env)
#   P19: (empty)  Accept default VITE_API_URL          (webui.py _setup_env)
#
# Notes:
#   - Config exists after P1-P2, so _download_coin_or_solvers() skips
#     "Use this location?" — it reads from config
#   - Gurobi: gurobipy not installed → _setup_gurobi() prints info, no prompt
#   - DB doesn't exist on fresh machine → _handle_existing_db() creates it, no prompt
#   - simple_knapsack and river_pollution_problem have no params → no extra prompts

echo "  Running unified wizard with 19 piped prompts..."

wizard_output=$(printf '%s\n' \
    1 y \
    y \
    1 y \
    y \
    1 '' analyst analyst test 3 \
    1 10 3 \
    y \
    1 '' '' \
    | uv run desdeo-setup 2>&1)

echo "$wizard_output"

# ── Phase 3: Verify All Artifacts ──────────────────────────────────────────

phase_header 3 "Verify All Artifacts"

# -- Config --
assert_file_exists /app/.desdeo/config.toml "config.toml created by wizard"

# -- Solvers --
SOLVER_DIR="$HOME/.local/share/desdeo/solvers"
export PATH="$SOLVER_DIR:$PATH"

assert_on_path bonmin "bonmin on PATH after wizard"
assert_on_path ipopt  "ipopt on PATH after wizard"
assert_on_path cbc    "cbc on PATH after wizard"

if grep -q "DESDEO solvers" ~/.bashrc 2>/dev/null; then
    pass "~/.bashrc contains solver PATH entry"
else
    fail "~/.bashrc missing solver PATH entry"
fi

# -- Database --
assert_file_exists /app/desdeo/api/test.db "database file created by wizard"

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

# -- WebUI / Node --
export NVM_DIR="$HOME/.nvm"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

assert_on_path node "Node.js installed by wizard"
assert_on_path npm  "npm available after wizard"

assert_file_exists /app/webui/node_modules "webui/node_modules created"
assert_file_exists /app/webui/.env         "webui/.env created"

env_content=$(cat /app/webui/.env)
assert_contains "$env_content" "API_BASE_URL" ".env has API_BASE_URL"
assert_contains "$env_content" "VITE_API_URL" ".env has VITE_API_URL"

# -- WebUI build --
echo "  Running npm run build..."
build_output=$(cd /app/webui && npm run build 2>&1) && \
    pass "npm run build succeeded" || \
    fail "npm run build failed"
echo "$build_output" | tail -5

# -- API startup --
echo "  Starting API server..."
uv run uvicorn desdeo.api.app:app --host 127.0.0.1 --port 8000 &
API_PID=$!

api_up=false
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/docs > /dev/null 2>&1; then
        api_up=true
        break
    fi
    sleep 1
done

if $api_up; then
    pass "API responds on :8000/docs"
else
    fail "API not responding on :8000/docs after 30s"
fi

kill $API_PID 2>/dev/null || true
wait $API_PID 2>/dev/null || true

# -- desdeo-setup check --
check_output=$(uv run desdeo-setup check 2>&1) || true
echo "$check_output"
assert_contains "$check_output" "Everything looks good" "desdeo-setup check reports all green"

# ── Phase 4: Fullstack Smoke Test (optional) ───────────────────────────────

phase_header 4 "Fullstack Smoke Test"

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

# ── Phase 5: Idempotent Re-run ─────────────────────────────────────────────

phase_header 5 "Idempotent Re-run"

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
