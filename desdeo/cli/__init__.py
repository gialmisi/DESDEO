"""DESDEO interactive setup CLI."""

import typer

app = typer.Typer(
    name="desdeo-setup",
    help="DESDEO interactive setup wizard.",
    invoke_without_command=True,
    no_args_is_help=False,
)

# Register subcommands
from desdeo.cli.checks import checks_app  # noqa: E402
from desdeo.cli.database import db_app  # noqa: E402
from desdeo.cli.solvers import solvers_app  # noqa: E402
from desdeo.cli.webui import webui_app  # noqa: E402

app.add_typer(checks_app, name="check")
app.add_typer(solvers_app, name="solvers")
app.add_typer(db_app, name="db")
app.add_typer(webui_app, name="webui")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """DESDEO interactive setup wizard."""
    if ctx.invoked_subcommand is None:
        # Default: run the full wizard
        from desdeo.cli.wizard import setup

        setup()
