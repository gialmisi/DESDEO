# Testing the DESDEO CLI in Docker

This guide explains how to build and run the DESDEO CLI end-to-end tests inside Docker. The Docker environment simulates a **fresh machine** with only Python 3.12 and `uv` installed — no Node.js, no solvers, no database. The test then exercises every phase of the setup wizard and verifies the results.

## Prerequisites

- **Docker** installed and running (`docker --version` to verify)
- On Apple Silicon (M1/M2/M3) or ARM machines, Docker Desktop with Rosetta or QEMU emulation enabled (the image targets `linux/amd64`)

## Quick Start

From the repository root:

```bash
make test-cli-e2e
```

This single command builds the Docker image and runs the full E2E test suite. If all assertions pass, you'll see `ALL PASSED` at the end.

## Step-by-Step

### 1. Build the Docker image

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.cli-test -t desdeo-cli-test .
```

This creates a `desdeo-cli-test` image based on `python:3.12-slim` with:

- System tools: `curl`, `git`, `build-essential`, `sqlite3`
- `uv` (Python package manager)
- All DESDEO Python dependencies (via `uv sync`)
- A clean workspace: no `node_modules`, no `.env`, no `test.db`

### 2. Run the automated E2E test

```bash
docker run --rm --platform linux/amd64 desdeo-cli-test
```

The container runs `docker/e2e_cli_test.sh`, which executes these phases:

| Phase | What it does |
|-------|-------------|
| 1     | **Verify Fresh State** — confirms no solvers, Node.js, or database exist |
| 1.5   | **Install Config** — creates `.desdeo/config.toml` with default paths |
| 2     | **Solver Setup** — downloads real COIN-OR solver binaries from GitHub |
| 3     | **WebUI Setup** — installs Node.js 24 via nvm, runs `npm install` |
| 4     | **Database Setup** — creates SQLite database with test users and problems |
| 5     | **Full Re-check** — runs `desdeo-setup check` and verifies all green |
| 6     | **Fullstack Smoke Test** — skipped unless `RUN_FULLSTACK=1` (see below) |
| 7     | **Idempotent Re-run** — reruns the wizard, confirms it detects nothing to do |

Each phase uses assertions (file exists, command on PATH, output contains string) and prints pass/fail results.

### 3. Run with fullstack smoke test (optional)

To also verify that the backend (uvicorn) and frontend (vite) start and respond to HTTP requests:

```bash
make test-cli-e2e-fullstack
```

Or manually:

```bash
docker run --rm --platform linux/amd64 -e RUN_FULLSTACK=1 desdeo-cli-test
```

This adds Phase 6, which starts both servers inside the container and checks that `localhost:8000` and `localhost:5173` respond.

## Full Wizard E2E Test

The standard `test-cli-e2e` target runs each subcommand (`desdeo-setup solvers`, `desdeo-setup webui`, `desdeo-setup db`) individually. The **wizard test** instead runs a single `desdeo-setup` invocation with all 19 prompts piped in sequence — exactly how a real user would experience the unified wizard.

```bash
make test-cli-wizard
```

This runs `docker/e2e_wizard_test.sh`, which:

1. **Verifies fresh state** — no solvers, Node.js, database, or config
2. **Runs the full wizard** in one `desdeo-setup` invocation with piped input covering install paths, solver download, database creation (SQLite + analyst user + default problems), and WebUI setup (nvm + npm install)
3. **Verifies all artifacts** — solvers on PATH, `.bashrc` updated, database with analyst user, Node/npm available, `webui/.env` and `node_modules` present, `desdeo-setup check` reports all green
4. **Fullstack smoke test** — skipped unless `RUN_FULLSTACK=1`
5. **Idempotent re-run** — reruns the wizard, confirms it detects nothing to do

To include the fullstack smoke test:

```bash
make test-cli-wizard-fullstack
```

## Interactive Debugging

To explore the test environment manually (e.g., run the wizard yourself, inspect files, debug failures):

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.cli-test -t desdeo-cli-test .
docker run --rm -it --platform linux/amd64 --entrypoint /bin/bash desdeo-cli-test
```

You are now inside the container at `/app`. Some things to try:

```bash
# Run the setup wizard interactively
uv run desdeo-setup

# Run just the environment check
uv run desdeo-setup check

# Run just solver setup
uv run desdeo-setup solvers

# Run just webui setup
uv run desdeo-setup webui

# Run just database setup
uv run desdeo-setup db

# Inspect the config file after running the wizard
cat .desdeo/config.toml

# Run the E2E test script manually
bash docker/e2e_cli_test.sh
```

### Testing install modes interactively

The wizard now asks where to install tools. When running interactively, you can try each mode:

1. **Default** — installs to `~/.local/share/desdeo/solvers` and `~/.nvm`
2. **Custom directory** — you specify a base path (e.g., `/opt/desdeo`)
3. **Project-local** — installs under `.desdeo/` in the repo (`.desdeo/solvers/`, `.desdeo/nvm/`)

Example testing project-local mode:

```bash
docker run --rm -it --platform linux/amd64 --entrypoint /bin/bash desdeo-cli-test

# Inside the container:
uv run desdeo-setup
# Pick option 3 (project-local), then y (apply to all)
# Proceed through solver download, webui setup, database setup
# Verify:
ls .desdeo/solvers/    # should contain bonmin, ipopt, cbc
ls .desdeo/nvm/        # should contain nvm.sh and node versions
cat .desdeo/config.toml
```

## Running Unit Tests

The config module has its own unit tests that run without Docker:

```bash
uv run pytest tests/test_cli_config.py -v
```

These tests use `tmp_path` and `monkeypatch` fixtures, so they don't touch real filesystem locations or require any external tools.

## Troubleshooting

**Build fails with network errors**
The image downloads `uv`, Python packages, solver binaries, nvm, and Node.js. Ensure you have a working internet connection and no corporate proxy blocking GitHub or npmjs.org.

**Tests fail on ARM/Apple Silicon**
The solver binaries are x86_64 Linux ELF executables. Use `--platform linux/amd64` (included in the Makefile targets) so Docker uses emulation.

**"Permission denied" on `e2e_cli_test.sh`**
The Dockerfile runs `chmod +x docker/e2e_cli_test.sh`. If building outside Docker, run `chmod +x docker/e2e_cli_test.sh` locally.

**Phase 2 (solvers) download is slow or times out**
The solver archive (~17 MB) is downloaded from GitHub Releases. Behind a slow connection, the download may take a while. There is no timeout on the download itself.

**Fullstack test fails (Phase 6)**
This phase starts real servers and waits 15 seconds for them to be ready. On slower machines or under heavy emulation, you may need to increase the `sleep 15` in `docker/e2e_cli_test.sh`.
