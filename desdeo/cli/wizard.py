"""Full setup wizard orchestrator — runs all phases in sequence."""

from __future__ import annotations

import typer

from desdeo.cli.styles import console, print_summary_panel, step_header, success


def _configure_install_paths() -> None:
    """Prompt the user for install directory preferences and persist them."""
    from desdeo.cli.config import (
        InstallMode,
        config_exists,
        get_project_root,
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

    choice = typer.prompt("  Choice", default="1")

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


def setup() -> None:
    """Run the full DESDEO setup wizard."""
    from desdeo.cli.checks import display_status, run_all_checks

    console.print("\n[bold]DESDEO Setup Wizard[/bold]")
    console.print("[dim]Scanning environment...[/dim]\n")

    status = run_all_checks()
    display_status(status)

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

    # Phase: Database
    if needs_db:
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
