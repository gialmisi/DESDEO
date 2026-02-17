"""Environment detection: Python, uv, Node, npm, solvers, Gurobi."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass

import typer

from desdeo.cli.styles import console, print_status_table, success, warn

checks_app = typer.Typer(help="Check the development environment.")


@dataclass
class CheckResult:
    """Result of a single environment check."""

    name: str
    ok: bool
    version: str = ""
    detail: str = ""


@dataclass
class EnvironmentStatus:
    """Aggregated environment check results."""

    python: CheckResult
    uv: CheckResult
    node: CheckResult
    nvm: CheckResult
    npm: CheckResult
    bonmin: CheckResult
    ipopt: CheckResult
    cbc: CheckResult
    gurobipy: CheckResult
    gurobi_license: CheckResult
    database: CheckResult
    webui: CheckResult

    def all_checks(self) -> list[CheckResult]:
        """Return all checks as a list."""
        return [
            self.python,
            self.uv,
            self.node,
            self.nvm,
            self.npm,
            self.bonmin,
            self.ipopt,
            self.cbc,
            self.gurobipy,
            self.gurobi_license,
            self.database,
            self.webui,
        ]

    @property
    def solvers_ok(self) -> bool:
        """True if all COIN-OR solvers are found."""
        return self.bonmin.ok and self.ipopt.ok and self.cbc.ok

    @property
    def everything_ok(self) -> bool:
        """True if all actionable components are set up.

        Node.js version is a warning (handled inside webui phase), not a blocker.
        """
        optional = {"nvm", "gurobipy", "Gurobi license", "Node.js"}
        return all(c.ok for c in self.all_checks() if c.name not in optional)


def _run_version(cmd: str) -> str | None:
    """Run `cmd --version` and return the first line, or None on failure."""
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass
    return None


def check_python() -> CheckResult:
    """Check Python version."""
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    ok = v >= (3, 12)
    detail = "" if ok else "Requires >= 3.12"
    return CheckResult(name="Python", ok=ok, version=version, detail=detail)


def check_uv() -> CheckResult:
    """Check if uv is available."""
    path = shutil.which("uv")
    if path:
        version = _run_version("uv") or "found"
        return CheckResult(name="uv", ok=True, version=version)
    return CheckResult(
        name="uv", ok=False, detail="Not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def check_node() -> CheckResult:
    """Check Node.js version."""
    version_str = _run_version("node")
    if not version_str:
        return CheckResult(name="Node.js", ok=False, detail="Not found")
    # Parse version like "v24.1.0" or "v18.0.0"
    try:
        clean = version_str.lstrip("v").split(".")[0]
        major = int(clean)
    except (ValueError, IndexError):
        return CheckResult(name="Node.js", ok=False, version=version_str, detail="Could not parse version")

    if major >= 24:
        return CheckResult(name="Node.js", ok=True, version=version_str)
    detail = f"Version {major} found, >= 24 recommended. Use 'nvm use 24' if available."
    return CheckResult(name="Node.js", ok=False, version=version_str, detail=detail)


def check_nvm() -> CheckResult:
    """Check if nvm is available."""
    import os
    from pathlib import Path

    nvm_dir = os.environ.get("NVM_DIR")
    if nvm_dir:
        return CheckResult(name="nvm", ok=True, version=nvm_dir, detail="optional")

    # Check config-specified nvm directory
    from desdeo.cli.config import config_exists, get_nvm_dir

    if config_exists():
        cfg_nvm = get_nvm_dir()
        if Path(cfg_nvm).is_dir():
            return CheckResult(name="nvm", ok=True, version=cfg_nvm, detail="optional (from config)")

    # Check common install location
    home = Path.home()
    if (home / ".nvm").is_dir():
        return CheckResult(name="nvm", ok=True, version=f"{home}/.nvm", detail="optional")
    return CheckResult(name="nvm", ok=False, detail="Not found (optional)")


def check_npm() -> CheckResult:
    """Check if npm is available."""
    version_str = _run_version("npm")
    if version_str:
        return CheckResult(name="npm", ok=True, version=version_str)
    return CheckResult(name="npm", ok=False, detail="Not found")


def check_solver(name: str) -> CheckResult:
    """Check if a solver binary is on PATH."""
    path = shutil.which(name)
    if path:
        return CheckResult(name=name, ok=True, version=path)
    return CheckResult(name=name, ok=False, detail="Not found on PATH")


def check_gurobipy() -> CheckResult:
    """Check if gurobipy is importable."""
    spec = importlib.util.find_spec("gurobipy")
    if spec:
        return CheckResult(name="gurobipy", ok=True, version="importable")
    return CheckResult(name="gurobipy", ok=False, detail="Not installed (optional)")


def check_gurobi_license() -> CheckResult:
    """Check if Gurobi has a valid license."""
    try:
        import gurobipy as gp

        m = gp.Model()
        m.dispose()
        return CheckResult(name="Gurobi license", ok=True, version="valid")
    except Exception as e:
        msg = str(e)
        if "gurobipy" in msg.lower() or "import" in msg.lower():
            return CheckResult(name="Gurobi license", ok=False, detail="gurobipy not installed")
        return CheckResult(name="Gurobi license", ok=False, detail="No valid license")


def check_database() -> CheckResult:
    """Check if the database exists and has tables."""
    from pathlib import Path

    api_dir = Path(__file__).resolve().parent.parent / "api"
    db_url = f"sqlite:///{api_dir / 'test.db'}"

    if db_url.startswith("sqlite"):
        db_file = db_url.replace("sqlite:///", "")
        path = Path(db_file)
        if path.exists():
            # Check if it has users
            try:
                from sqlmodel import Session, create_engine, select

                engine = create_engine(db_url, connect_args={"check_same_thread": False})
                import desdeo.api.models  # noqa: F401
                from desdeo.api.models import User

                with Session(engine) as session:
                    users = session.exec(select(User)).all()
                    if users:
                        names = ", ".join(u.username for u in users)
                        return CheckResult(name="Database", ok=True, version=db_file, detail=f"users: {names}")
                return CheckResult(name="Database", ok=False, version=db_file, detail="No users found")
            except Exception:
                return CheckResult(name="Database", ok=False, version=db_file, detail="Exists but could not query")
        return CheckResult(name="Database", ok=False, detail="Not created yet")
    return CheckResult(name="Database", ok=False, detail="Non-SQLite DB (check manually)")


def check_webui() -> CheckResult:
    """Check if the webui has node_modules and .env."""
    from pathlib import Path

    webui_dir = Path(__file__).resolve().parent.parent.parent / "webui"
    if not webui_dir.exists():
        return CheckResult(name="WebUI", ok=False, detail="webui/ directory not found")

    has_modules = (webui_dir / "node_modules").is_dir()
    has_env = (webui_dir / ".env").is_file()

    if has_modules and has_env:
        return CheckResult(name="WebUI", ok=True, detail="node_modules + .env present")
    parts = []
    if not has_modules:
        parts.append("npm install needed")
    if not has_env:
        parts.append(".env missing")
    return CheckResult(name="WebUI", ok=False, detail=", ".join(parts))


def run_all_checks() -> EnvironmentStatus:
    """Run all environment checks and return aggregated status."""
    return EnvironmentStatus(
        python=check_python(),
        uv=check_uv(),
        node=check_node(),
        nvm=check_nvm(),
        npm=check_npm(),
        bonmin=check_solver("bonmin"),
        ipopt=check_solver("ipopt"),
        cbc=check_solver("cbc"),
        gurobipy=check_gurobipy(),
        gurobi_license=check_gurobi_license(),
        database=check_database(),
        webui=check_webui(),
    )


def display_status(status: EnvironmentStatus) -> None:
    """Display environment status as a table."""
    rows: list[tuple[str, str, str]] = []
    for c in status.all_checks():
        icon = "[green]\u2713[/green]" if c.ok else "[red]\u2717[/red]"
        version = c.version if c.version else ""
        detail = c.detail if c.detail else ""
        label = f"{version}  {detail}".strip() if version or detail else ""
        rows.append((c.name, icon, label))
    print_status_table(rows)


@checks_app.callback(invoke_without_command=True)
def check() -> None:
    """Check the DESDEO development environment."""
    console.print("\n[bold]Environment Check[/bold]\n")
    status = run_all_checks()
    display_status(status)

    # Summary
    console.print()
    if status.everything_ok:
        success("Everything looks good!")
    else:
        optional = {"nvm", "gurobipy", "Gurobi license"}
        required_fails = [c.name for c in status.all_checks() if not c.ok and c.name not in optional]
        if required_fails:
            warn(f"Issues found: {', '.join(required_fails)}")
            console.print("  Run [bold]desdeo-setup[/bold] for guided setup.\n")
