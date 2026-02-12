"""Full setup wizard orchestrator — runs all phases in sequence."""

from __future__ import annotations

import typer

from desdeo.cli.styles import console, print_summary_panel, step_header, success


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
        if typer.confirm("Proceed with solver setup?", default=True):
            step_header(step, total, "Solver Setup")
            from desdeo.cli.solvers import solvers

            solvers()
        step += 1

    # Phase: Database
    if needs_db:
        if typer.confirm("Proceed with database setup?", default=True):
            step_header(step, total, "Database Setup")
            from desdeo.cli.database import db

            db()
        step += 1

    # Phase: WebUI
    if needs_webui:
        if typer.confirm("Proceed with WebUI setup?", default=True):
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
