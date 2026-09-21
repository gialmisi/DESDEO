# This justfile defines several recipes to run tests and other useful scripts.
#
# To execute a recipe, issue the command "just <recipe>". Requires `just`,
# which is installed automatically as a dev dependency via `uv sync`
# (PyPI package: rust-just).
#
# Run "just --list" to see all available recipes.

# Default recipe: list available recipes
default:
    @just --list

# Pytest configuration (can be overridden, e.g., `just test PYTEST_SKIP=""`)

PYTEST := "pytest -n auto"
PYTEST_SKIP := '-m "not fixme"'
PYTEST_OPTS := "--disable-warnings"
TEST_API_PATH := "./desdeo/api/tests"

# Run the typical tests, skipping tests marked to be skipped.
test:
    {{ PYTEST }} {{ PYTEST_SKIP }} {{ PYTEST_OPTS }}

# Run only the API tests.
test-api:
    {{ PYTEST }} {{ PYTEST_SKIP }} {{ PYTEST_OPTS }} {{ TEST_API_PATH }}

# Run all tests regardless of marks. This can be very slow.
test-all:
    {{ PYTEST }}

# Run only necessary tests given the changes in the code (pytest-testmon).
# NOTE: testmon must be run WITHOUT xdist (`-n auto`) and WITHOUT a `-m` mark
# expression. Either one stops testmon from saving a stable baseline, so it
# silently reruns the whole suite every time. Hence no PYTEST/PYTEST_SKIP here.
# Instead, TESTMON_SKIP_FIXME makes the root conftest skip `fixme` tests (the
# testmon-friendly equivalent of `-m "not fixme"`).
test-changes:
    TESTMON_SKIP_FIXME=1 pytest --testmon {{ PYTEST_OPTS }}

# Rerun the last failures only.
test-failures:
    {{ PYTEST }} --lf {{ PYTEST_SKIP }} {{ PYTEST_OPTS }}

# Run the web UI unit tests (vitest).
test-webui:
    cd webui && npx vitest run --config vitest.config.ts

# Run all tests (Python + WebUI).
test-everything: test test-webui

# Run the web-API and web-GUI for local development.
fullstack:
    python run_fullstack.py

# Serve docs locally (fast rebuild).
docs-fast:
    mkdocs serve -f mkdocs.yml

# Serve docs locally (ReadTheDocs config).
docs-rtd:
    mkdocs serve -f mkdocs.rtd.yml

# Run pre-commit hooks on staged files.
lint:
    pre-commit run

# run pre-commit hooks on all files.
lint-all:
    pre-commit run --all-files

# Runs from desdeo/api, because that is where the API server looks for test.db.
# Seed the cat and dog breed problems (WARNING: drops every table in test.db).
demo-catsanddogs-db:
    cd desdeo/api && python db_init_catsanddogs.py

# Run the cats and dogs demo (seed it first with `just demo-catsanddogs-db`).
demo-catsanddogs:
    @echo "Log in as analyst / analyst, then open the demo at:"
    @echo "  http://localhost:5173/demos/cats-and-dogs"
    @echo "(Vite picks another port if 5173 is taken, see its output below.)"
    @echo ""
    python run_fullstack.py
