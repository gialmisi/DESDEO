"""Full setup wizard orchestrator — runs all phases in sequence."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from desdeo.cli.styles import console, print_summary_panel, step_header, success


def _configure_install_paths() -> None:
    """Prompt the user for install directory preferences and persist them."""
    from desdeo.cli.config import (
        InstallMode,
        config_exists,
        get_project_root,
        is_conda_env,
        load_config,
        resolve_paths_for_mode,
        save_config,
    )

    # If config already exists, offer to keep it
    if config_exists():
        cfg = load_config()
        console.print(f"  [bold]Install paths configured:[/bold] mode={cfg['mode']}")
        console.print(f"    solvers : {cfg['solvers']}")
        console.print(f"    nvm_dir : {cfg['nvm_dir']}")
        if not typer.confirm("  Reconfigure install paths?", default=False):
            return

    root = get_project_root()
    default_paths = resolve_paths_for_mode(InstallMode.DEFAULT)
    local_paths = resolve_paths_for_mode(InstallMode.PROJECT_LOCAL)

    console.print("\n  [bold]Install Location[/bold]")
    console.print("    Where should CLI-managed tools be installed?\n")
    console.print(f"    1) Default locations")
    console.print(f"       solvers : {default_paths['solvers']}")
    console.print(f"       nvm     : {default_paths['nvm_dir']}")
    console.print(f"    2) Custom directory (you specify a base path)")
    console.print(f"    3) Project-local (.desdeo/ inside the repo)")
    console.print(f"       solvers : {local_paths['solvers']}")
    console.print(f"       nvm     : {local_paths['nvm_dir']}")
    console.print()

    default_choice = "3" if is_conda_env() else "1"
    choice = typer.prompt("  Choice", default=default_choice)

    if choice == "2":
        custom_base = typer.prompt("  Base directory for tools")
        paths = resolve_paths_for_mode(InstallMode.CUSTOM, custom_base)
        mode = InstallMode.CUSTOM
    elif choice == "3":
        paths = local_paths
        mode = InstallMode.PROJECT_LOCAL
        custom_base = None
    else:
        paths = default_paths
        mode = InstallMode.DEFAULT
        custom_base = None

    # Per-component overrides?
    console.print(f"\n    solvers : {paths['solvers']}")
    console.print(f"    nvm     : {paths['nvm_dir']}\n")

    if typer.confirm("  Apply to all components?", default=True):
        save_config(
            mode=mode,
            solvers=paths["solvers"],
            nvm_dir=paths["nvm_dir"],
            custom_base=custom_base if choice == "2" else None,
        )
    else:
        solver_dir = typer.prompt("  Solver directory", default=paths["solvers"])
        nvm_dir = typer.prompt("  nvm directory", default=paths["nvm_dir"])
        save_config(
            mode=mode,
            solvers=solver_dir,
            nvm_dir=nvm_dir,
            custom_base=custom_base if choice == "2" else None,
        )

    success("Install paths saved to .desdeo/config.toml")


def _dependency_group_phase() -> dict[str, bool]:
    """Detect missing dependency groups and offer to install them.

    Returns a dict ``{group_name: is_available}`` for every group found in
    pyproject.toml.
    """
    from desdeo.cli.config import check_dependency_group, ensure_dependency_groups, read_dependency_groups
    from desdeo.cli.styles import info, warn

    groups = read_dependency_groups()
    if not groups:
        info("Could not locate pyproject.toml — skipping dependency-group check.")
        return {}

    # Check each group
    group_missing: dict[str, list[str]] = {}
    for name in groups:
        missing, _ = check_dependency_group(name)
        group_missing[name] = missing

    any_missing = any(m for m in group_missing.values())
    if not any_missing:
        return {name: True for name in groups}

    console.print("  [bold]Dependency Groups[/bold]")
    console.print("    Some optional dependency groups have missing packages:\n")
    for name, missing in group_missing.items():
        if missing:
            preview = ", ".join(m.split("[")[0].split(">")[0].split("<")[0].strip() for m in missing[:3])
            if len(missing) > 3:
                preview += ", ..."
            console.print(f"      {name:12s} {len(missing)} missing ({preview})")
        else:
            console.print(f"      {name:12s} [green]OK[/green]")

    console.print()
    console.print("    1) Install all development groups (recommended)")
    console.print("    2) Select specific groups")
    console.print("    3) Skip\n")

    choice = typer.prompt("  Choice", default="1")

    # Groups included in "all-dev" (the standard development set)
    all_dev_groups = ["dev", "docs", "jupyter", "web", "viz", "tools"]

    if choice == "1":
        # Install all dev groups that have missing packages
        to_install = [g for g in all_dev_groups if g in group_missing and group_missing[g]]
        if to_install:
            return ensure_dependency_groups(to_install)
    elif choice == "2":
        # Let user pick
        available = [name for name, m in group_missing.items() if m]
        console.print("    Select groups to install (comma-separated numbers):\n")
        for i, name in enumerate(available, 1):
            console.print(f"      {i}) {name} ({len(group_missing[name])} missing)")
        console.print()
        selection = typer.prompt("  Groups", default=",".join(str(i) for i in range(1, len(available) + 1)))
        indices = [int(s.strip()) - 1 for s in selection.split(",") if s.strip().isdigit()]
        to_install = [available[i] for i in indices if 0 <= i < len(available)]
        if to_install:
            return ensure_dependency_groups(to_install)
    else:
        warn("Skipping dependency group installation.")

    # Return current status (re-check)
    result: dict[str, bool] = {}
    for name in groups:
        missing, _ = check_dependency_group(name)
        result[name] = len(missing) == 0
    return result


def setup() -> None:
    """Run the full DESDEO setup wizard."""
    from desdeo.cli.checks import display_status, run_all_checks

    console.print("\n[bold]DESDEO Setup Wizard[/bold]")
    console.print("[dim]Scanning environment...[/dim]\n")

    # Phase 0: dependency group installation (before any checks)
    group_status = _dependency_group_phase()
    console.print()

    status = run_all_checks()
    display_status(status)

    # Detect conda environment
    from desdeo.cli.config import is_conda_env

    if is_conda_env():
        conda_name = Path(os.environ["CONDA_PREFIX"]).name
        console.print(f"\n  [bold]Conda environment detected:[/bold] {conda_name}")
        console.print("    Solver PATH will use conda activation scripts.")
        console.print("    Node.js will be installed via conda if needed.")

    if status.everything_ok:
        console.print()
        success("Everything is already set up!")
        _print_final_summary(status)
        return

    # Configure install paths (before any phases run)
    _configure_install_paths()

    # Determine which phases need work
    needs_solvers = not status.solvers_ok
    needs_db = not status.database.ok
    needs_webui = not status.webui.ok

    console.print()
    phases: list[str] = []
    if needs_solvers:
        phases.append("solvers")
    if needs_db:
        phases.append("database")
    if needs_webui:
        phases.append("webui")

    console.print(f"  Setup needed: [bold]{', '.join(phases)}[/bold]\n")

    total = len(phases)
    step = 1

    # Phase: Solvers
    if needs_solvers:
        missing = [n for n in ("bonmin", "ipopt", "cbc") if not getattr(status, n).ok]
        console.print(f"  [bold]Solver Setup[/bold] — will download COIN-OR binaries ({', '.join(missing)})")
        console.print("    Downloads ~17 MB from GitHub, installs to ~/.local/share/desdeo/solvers,")
        console.print("    and adds the directory to your PATH.\n")
        if typer.confirm("Proceed?", default=True):
            step_header(step, total, "Solver Setup")
            from desdeo.cli.solvers import solvers

            solvers()
        step += 1

    # Phase: Database (requires 'web' dependency group for sqlmodel etc.)
    if needs_db:
        web_available = group_status.get("web", False)
        if not web_available:
            from desdeo.cli.styles import warn

            warn("Skipping database setup — 'web' dependency group is not installed.")
            warn("Run desdeo-setup again after installing the web dependencies.")
        else:
            console.print("  [bold]Database Setup[/bold] — will create a local SQLite database")
            console.print("    Creates desdeo/api/test.db, sets up user accounts (analyst + optional")
            console.print("    decision makers), and seeds optimization test problems.\n")
            if typer.confirm("Proceed?", default=True):
                step_header(step, total, "Database Setup")
                from desdeo.cli.database import db

                db()
        step += 1

    # Phase: WebUI
    if needs_webui:
        console.print("  [bold]WebUI Setup[/bold] — will install frontend dependencies and configure .env")
        console.print("    Installs Node.js 24 via nvm if needed, runs npm install in webui/")
        console.print("    (~770 packages), and writes API endpoint URLs to webui/.env.\n")
        if typer.confirm("Proceed?", default=True):
            step_header(step, total, "WebUI Setup")
            from desdeo.cli.webui import webui

            webui()
        step += 1

    # Re-check and print final summary
    status = run_all_checks()
    _print_final_summary(status)


def _print_final_summary(status) -> None:
    """Print the final setup summary panel."""
    from desdeo.cli.checks import check_solver

    solver_parts = []
    for name in ["bonmin", "ipopt", "cbc"]:
        s = check_solver(name)
        solver_parts.append(f"{name} \u2713" if s.ok else f"{name} \u2717")

    db_status = status.database.detail if status.database.ok else "not configured"
    webui_status = "ready" if status.webui.ok else "not configured"

    summary_lines = [
        f"  Python:    {status.python.version}",
        f"  Solvers:   {' '.join(solver_parts)}",
        f"  Database:  {db_status}",
        f"  WebUI:     {webui_status}",
        "",
        "  Start:  make fullstack",
        "  Open:   http://localhost:5173",
        "  Tests:  make test",
    ]
    console.print()
    print_summary_panel(summary_lines)
