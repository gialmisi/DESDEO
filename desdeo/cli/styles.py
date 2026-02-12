"""Shared Rich console, formatting helpers, and problem/solver catalogs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    pass

console = Console()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def step_header(n: int, total: int, title: str) -> None:
    """Print a styled step header like '[2/5] Database setup...'."""
    console.print(f"\n[bold cyan][{n}/{total}][/bold cyan] [bold]{title}[/bold]")


def success(msg: str) -> None:
    """Print a success message."""
    console.print(f"  [green]\u2713[/green] {msg}")


def warn(msg: str) -> None:
    """Print a warning message."""
    console.print(f"  [yellow]\u26a0[/yellow] {msg}")


def fail(msg: str) -> None:
    """Print a failure message."""
    console.print(f"  [red]\u2717[/red] {msg}")


def info(msg: str) -> None:
    """Print an info message."""
    console.print(f"  [dim]{msg}[/dim]")


def print_summary_panel(lines: list[str], title: str = "DESDEO Setup Complete!") -> None:
    """Print a bordered summary panel."""
    body = "\n".join(lines)
    console.print(Panel(body, title=f"[bold green]{title}[/bold green]", border_style="green", padding=(1, 2)))


def print_status_table(rows: list[tuple[str, str, str]]) -> None:
    """Print a status table with columns: Component, Status, Detail."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    for name, status, detail in rows:
        table.add_row(name, status, detail)
    console.print(table)


# ---------------------------------------------------------------------------
# Solver download URLs
# ---------------------------------------------------------------------------

SOLVER_URLS: dict[tuple[str, str], str] = {
    ("linux", "x86_64"): (
        "https://github.com/industrial-optimization-group/DESDEO/releases/download/supplementary/solver_binaries.tgz"
    ),
    ("darwin", "x86_64"): (
        "https://github.com/industrial-optimization-group/DESDEO/releases/"
        "download/supplementary/coin.macos64.20211124.tgz"
    ),
    ("darwin", "arm64"): (
        "https://github.com/industrial-optimization-group/DESDEO/releases/"
        "download/supplementary/coin.macos64.20211124.tgz"
    ),
    ("win32", "amd64"): (
        "https://github.com/industrial-optimization-group/DESDEO/releases/"
        "download/supplementary/coin.mswin64.20230221.zip"
    ),
}

AMPL_URL = "https://ampl.com/products/solvers/open-source-solvers/"

GUROBI_ACADEMIC_URL = "https://www.gurobi.com/academia/academic-program-and-licenses/"

DEFAULT_SOLVER_DIR_UNIX = "~/.local/share/desdeo/solvers"
DEFAULT_SOLVER_DIR_WIN = "%LOCALAPPDATA%\\desdeo\\solvers"


def get_default_solver_dir() -> str:
    """Return the default solver install directory for the current platform."""
    if sys.platform == "win32":
        return DEFAULT_SOLVER_DIR_WIN
    return DEFAULT_SOLVER_DIR_UNIX


# ---------------------------------------------------------------------------
# Test problem catalog
# ---------------------------------------------------------------------------


@dataclass
class TestProblemEntry:
    """Metadata for a test problem factory function."""

    name: str
    module: str
    category: str
    description: str
    params: list[tuple[str, int]] = field(default_factory=list)  # (param_name, default_value)


# All test problems from desdeo.problem.testproblems, organized by category.
TEST_PROBLEM_CATALOG: list[TestProblemEntry] = [
    # --- Benchmark ---
    TestProblemEntry(
        name="dtlz2",
        module="desdeo.problem.testproblems",
        category="Benchmark",
        description="DTLZ2 scalable benchmark problem",
        params=[("n_variables", 10), ("n_objectives", 3)],
    ),
    TestProblemEntry(
        name="zdt1",
        module="desdeo.problem.testproblems",
        category="Benchmark",
        description="ZDT1 benchmark (2 objectives)",
        params=[("number_of_variables", 30)],
    ),
    TestProblemEntry(
        name="zdt2",
        module="desdeo.problem.testproblems",
        category="Benchmark",
        description="ZDT2 benchmark (2 objectives)",
        params=[("n_variables", 30)],
    ),
    TestProblemEntry(
        name="zdt3",
        module="desdeo.problem.testproblems",
        category="Benchmark",
        description="ZDT3 benchmark (2 objectives)",
        params=[("n_variables", 30)],
    ),
    TestProblemEntry(
        name="binh_and_korn",
        module="desdeo.problem.testproblems",
        category="Benchmark",
        description="Binh and Korn two-objective problem",
    ),
    # --- Real-world ---
    TestProblemEntry(
        name="river_pollution_problem",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="River pollution problem (5 objectives)",
    ),
    TestProblemEntry(
        name="river_pollution_problem_discrete",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="River pollution (discrete representation)",
    ),
    TestProblemEntry(
        name="river_pollution_scenario",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="River pollution (scenario-based uncertainty)",
    ),
    TestProblemEntry(
        name="forest_problem_discrete",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Forest management problem (discrete)",
    ),
    TestProblemEntry(
        name="rocket_injector_design",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Rocket injector design problem",
    ),
    TestProblemEntry(
        name="spanish_sustainability_problem",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Spanish sustainability problem",
    ),
    TestProblemEntry(
        name="spanish_sustainability_problem_discrete",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Spanish sustainability (discrete)",
    ),
    TestProblemEntry(
        name="best_cake_problem",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Best cake recipe problem",
    ),
    # --- Engineering (RE problems) ---
    TestProblemEntry(
        name="re21",
        module="desdeo.problem.testproblems",
        category="Engineering",
        description="Real-world engineering problem RE21",
    ),
    TestProblemEntry(
        name="re22",
        module="desdeo.problem.testproblems",
        category="Engineering",
        description="Real-world engineering problem RE22",
    ),
    TestProblemEntry(
        name="re23",
        module="desdeo.problem.testproblems",
        category="Engineering",
        description="Real-world engineering problem RE23",
    ),
    TestProblemEntry(
        name="re24",
        module="desdeo.problem.testproblems",
        category="Engineering",
        description="Real-world engineering problem RE24",
    ),
    # --- Knapsack ---
    TestProblemEntry(
        name="simple_knapsack",
        module="desdeo.problem.testproblems",
        category="Combinatorial",
        description="Simple multiobjective knapsack",
    ),
    TestProblemEntry(
        name="simple_knapsack_vectors",
        module="desdeo.problem.testproblems",
        category="Combinatorial",
        description="Knapsack with vector variables",
    ),
    # --- Simple / Test ---
    TestProblemEntry(
        name="simple_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Simple test problem",
    ),
    TestProblemEntry(
        name="simple_linear_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Simple linear test problem",
    ),
    TestProblemEntry(
        name="simple_integer_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Simple integer test problem",
    ),
    TestProblemEntry(
        name="simple_data_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Simple data-based problem",
    ),
    TestProblemEntry(
        name="simple_scenario_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Simple scenario test problem",
    ),
    TestProblemEntry(
        name="nimbus_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="NIMBUS test problem",
    ),
    TestProblemEntry(
        name="pareto_navigator_test_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Pareto Navigator test problem",
    ),
    # simulator_problem requires external test fixtures (file_dir) — not available in setup context
    TestProblemEntry(
        name="multi_valued_constraint_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Multi-valued constraint problem",
    ),
    TestProblemEntry(
        name="mixed_variable_dimensions_problem",
        module="desdeo.problem.testproblems",
        category="Simple",
        description="Mixed variable dimensions problem",
    ),
    # --- MOMIP ---
    TestProblemEntry(
        name="momip_ti2",
        module="desdeo.problem.testproblems",
        category="MOMIP",
        description="MOMIP test instance TI2",
    ),
    TestProblemEntry(
        name="momip_ti7",
        module="desdeo.problem.testproblems",
        category="MOMIP",
        description="MOMIP test instance TI7",
    ),
    # --- MCWB ---
    TestProblemEntry(
        name="mcwb_solid_rectangular_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB solid rectangular beam",
    ),
    TestProblemEntry(
        name="mcwb_hollow_rectangular_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB hollow rectangular beam",
    ),
    TestProblemEntry(
        name="mcwb_equilateral_tbeam_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB equilateral T-beam",
    ),
    TestProblemEntry(
        name="mcwb_ragsdell1976_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB Ragsdell 1976 problem",
    ),
    TestProblemEntry(
        name="mcwb_square_channel_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB square channel",
    ),
    TestProblemEntry(
        name="mcwb_tapered_channel_problem",
        module="desdeo.problem.testproblems",
        category="MCWB",
        description="MCWB tapered channel",
    ),
    # --- Misc ---
    TestProblemEntry(
        name="dmitry_forest_problem_disc",
        module="desdeo.problem.testproblems",
        category="Real-world",
        description="Dmitry's forest problem (discrete)",
    ),
]


# Quick-access groups for the wizard
DEFAULT_PROBLEMS = ["dtlz2", "simple_knapsack", "river_pollution_problem"]

BENCHMARK_PROBLEMS = ["dtlz2", "zdt1", "zdt2", "zdt3", "binh_and_korn"]

REAL_WORLD_PROBLEMS = [
    "river_pollution_problem",
    "river_pollution_problem_discrete",
    "forest_problem_discrete",
    "rocket_injector_design",
    "spanish_sustainability_problem",
    "spanish_sustainability_problem_discrete",
    "best_cake_problem",
    "dmitry_forest_problem_disc",
]


def get_problem_entry(name: str) -> TestProblemEntry | None:
    """Look up a problem entry by name."""
    for entry in TEST_PROBLEM_CATALOG:
        if entry.name == name:
            return entry
    return None


def get_problems_by_category() -> dict[str, list[TestProblemEntry]]:
    """Group problems by category."""
    groups: dict[str, list[TestProblemEntry]] = {}
    for entry in TEST_PROBLEM_CATALOG:
        groups.setdefault(entry.category, []).append(entry)
    return groups
