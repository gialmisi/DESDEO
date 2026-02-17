"""Solver download, PATH setup, and Gurobi license guide."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import typer

from desdeo.cli.checks import check_gurobi_license, check_gurobipy, check_solver
from desdeo.cli.styles import (
    AMPL_URL,
    GUROBI_ACADEMIC_URL,
    SOLVER_URLS,
    console,
    fail,
    get_default_solver_dir,
    info,
    step_header,
    success,
    warn,
)

solvers_app = typer.Typer(help="Set up optimization solvers.")

COIN_OR_BINARIES = ["bonmin", "ipopt", "cbc"]


def _get_platform_key() -> tuple[str, str]:
    """Return (sys.platform, machine) normalized for solver URL lookup."""
    machine = platform.machine().lower()
    # Normalize machine names
    if machine in ("x86_64", "amd64"):
        machine = "x86_64" if sys.platform != "win32" else "amd64"
    return (sys.platform, machine)


def _detect_shell_rc() -> Path | None:
    """Detect the user's shell rc file."""
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        # Prefer .bashrc, fall back to .bash_profile
        bashrc = home / ".bashrc"
        if bashrc.exists():
            return bashrc
        return home / ".bash_profile"
    return None


def _add_to_path(solver_dir: Path) -> None:
    """Offer to add solver directory to PATH via shell rc file."""
    rc_file = _detect_shell_rc()
    if not rc_file:
        warn("Could not detect shell config file.")
        console.print("    Add this to your shell config manually:")
        console.print(f'    [bold]export PATH="{solver_dir}:$PATH"[/bold]\n')
        return

    line = f'\nexport PATH="{solver_dir}:$PATH"  # DESDEO solvers\n'

    # Check if already present
    if rc_file.exists():
        content = rc_file.read_text()
        if str(solver_dir) in content:
            info(f"PATH entry already exists in {rc_file}")
            return

    add = typer.confirm(f"  Add {solver_dir} to PATH in {rc_file}?", default=True)
    if add:
        with rc_file.open("a") as f:
            f.write(line)
        success(f"Added to {rc_file}")
        warn("Run 'source " + str(rc_file) + "' or open a new terminal for changes to take effect.")
    else:
        console.print("    Add this manually:")
        console.print(f'    [bold]export PATH="{solver_dir}:$PATH"[/bold]\n')


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a Rich progress bar."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=None)

        def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.update(task, total=total_size)
            progress.update(task, advance=block_size)

        urlretrieve(url, str(dest), reporthook=_reporthook)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tgz or .zip archive to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        # .tgz or .tar.gz
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)


def _find_binaries_in_dir(directory: Path) -> Path | None:
    """Find the directory containing solver binaries after extraction."""
    # Check if binaries are directly in the directory
    if (directory / "bonmin").exists() or (directory / "bonmin.exe").exists():
        return directory

    # Check subdirectories (archives often have a top-level folder)
    for child in directory.iterdir():
        if child.is_dir():
            if (child / "bonmin").exists() or (child / "bonmin.exe").exists():
                return child
            # One more level deep
            for grandchild in child.iterdir():
                if grandchild.is_dir() and ((grandchild / "bonmin").exists() or (grandchild / "bonmin.exe").exists()):
                    return grandchild
    return None


def _make_executable(directory: Path) -> None:
    """Make solver binaries executable (Unix only)."""
    if sys.platform == "win32":
        return
    for binary in COIN_OR_BINARIES:
        path = directory / binary
        if path.exists():
            path.chmod(0o755)


def _download_coin_or_solvers() -> None:
    """Download and install COIN-OR solvers from DESDEO GitHub releases."""
    from desdeo.cli.config import config_exists, get_solver_dir

    platform_key = _get_platform_key()
    url = SOLVER_URLS.get(platform_key)
    if not url:
        fail(f"No pre-compiled solvers available for {platform_key[0]} / {platform_key[1]}")
        console.print(f"    Please download manually from: {AMPL_URL}")
        return

    # Use config if available; otherwise fall back to interactive prompt
    if config_exists():
        solver_dir = Path(get_solver_dir())
        console.print(f"\n  Install location (from config): [bold]{solver_dir}[/bold]")
    else:
        solver_dir = Path(get_default_solver_dir()).expanduser()
        console.print(f"\n  Install location: [bold]{solver_dir}[/bold]")
        custom = typer.confirm("  Use this location?", default=True)
        if not custom:
            custom_path = typer.prompt("  Enter install directory")
            solver_dir = Path(custom_path).expanduser()

    solver_dir.mkdir(parents=True, exist_ok=True)

    # Download
    ext = ".zip" if platform_key[0] == "win32" else ".tgz"
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / f"solvers{ext}"
        console.print()
        _download_with_progress(url, archive_path)

        # Extract
        console.print("  Extracting...")
        extract_dir = Path(tmpdir) / "extracted"
        _extract_archive(archive_path, extract_dir)

        # Find binaries
        bin_dir = _find_binaries_in_dir(extract_dir)
        if not bin_dir:
            fail("Could not find solver binaries in the downloaded archive.")
            return

        # Copy binaries to solver_dir
        for item in bin_dir.iterdir():
            dest = solver_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)

    _make_executable(solver_dir)

    # Verify
    found = []
    for binary in COIN_OR_BINARIES:
        if (solver_dir / binary).exists() or (solver_dir / f"{binary}.exe").exists():
            found.append(binary)
    if found:
        success(f"Installed: {', '.join(found)} -> {solver_dir}")
    else:
        fail("Installation may have failed — binaries not found in target directory.")

    # Offer PATH addition
    if str(solver_dir) not in os.environ.get("PATH", ""):
        _add_to_path(solver_dir)


def _specify_existing_path() -> None:
    """Let user specify an existing solver directory."""
    path_str = typer.prompt("  Path to directory containing solver binaries")
    solver_dir = Path(path_str).expanduser()
    if not solver_dir.is_dir():
        fail(f"Directory not found: {solver_dir}")
        return

    found = []
    for binary in COIN_OR_BINARIES:
        if (solver_dir / binary).exists() or (solver_dir / f"{binary}.exe").exists():
            found.append(binary)

    if found:
        success(f"Found: {', '.join(found)} in {solver_dir}")
        if str(solver_dir) not in os.environ.get("PATH", ""):
            _add_to_path(solver_dir)
    else:
        fail(f"No solver binaries found in {solver_dir}")


def _setup_gurobi() -> None:
    """Guide user through Gurobi setup."""
    gp_check = check_gurobipy()
    lic_check = check_gurobi_license()

    if not gp_check.ok:
        console.print("\n  [dim]Gurobi is a commercial solver, optional for DESDEO.[/dim]")
        console.print("  [dim]Install with: pip install gurobipy[/dim]")
        console.print("  [dim]Free academic licenses are available.[/dim]")
        return

    if lic_check.ok:
        success("Gurobi is installed and licensed.")
        return

    console.print("\n  Gurobi is installed but [yellow]no valid license[/yellow] was found.\n")
    console.print("  For academic use, Gurobi offers free Named-User licenses:")
    console.print(f"    1) Open the license page ({GUROBI_ACADEMIC_URL})")
    console.print("    2) Register/log in with your academic email")
    console.print("    3) Generate a Named-User Academic license")
    console.print("    4) Copy the grbgetkey command (e.g., grbgetkey ae36ac20-...)")
    console.print("    5) Run it on this machine (must be on your institution's network)\n")

    choice = typer.prompt(
        "  Options: (1) Open license page  (2) Run grbgetkey  (3) Skip",
        default="3",
    )

    if choice == "1":
        import webbrowser

        webbrowser.open(GUROBI_ACADEMIC_URL)
        console.print("  Opened browser. Come back when you have a license key.")
        key_input = typer.prompt("  Enter grbgetkey UUID (or press Enter to skip)", default="")
        if key_input:
            subprocess.run(["grbgetkey", key_input])
    elif choice == "2":
        key_input = typer.prompt("  Enter grbgetkey UUID")
        subprocess.run(["grbgetkey", key_input])
    else:
        info("Skipping Gurobi license setup.")


@solvers_app.callback(invoke_without_command=True)
def solvers() -> None:
    """Set up optimization solvers (COIN-OR + Gurobi)."""
    step_header(1, 2, "COIN-OR Solvers (bonmin, ipopt, cbc)")

    # Check current status
    solver_status = {name: check_solver(name) for name in COIN_OR_BINARIES}
    all_found = all(s.ok for s in solver_status.values())

    if all_found:
        for name, s in solver_status.items():
            success(f"{name}: {s.version}")
        console.print("  All COIN-OR solvers found!\n")
    else:
        for name, s in solver_status.items():
            if s.ok:
                success(f"{name}: {s.version}")
            else:
                fail(f"{name}: not found")

        console.print("\n  Options:")
        console.print("    1) Download from DESDEO GitHub releases (recommended)")
        console.print(f"    2) Download from AMPL ({AMPL_URL})")
        console.print("    3) Skip")
        console.print("    4) Specify existing path\n")

        choice = typer.prompt("  Choice", default="1")

        if choice == "1":
            _download_coin_or_solvers()
        elif choice == "2":
            import webbrowser

            webbrowser.open(AMPL_URL)
            console.print("  Opened AMPL download page in browser.")
            console.print("  Download bonmin, ipopt, and cbc, then add them to your PATH.")
        elif choice == "4":
            _specify_existing_path()
        else:
            info("Skipping COIN-OR solver setup.")

    step_header(2, 2, "Gurobi (optional)")
    _setup_gurobi()

    console.print()
    info("Docker-based solver setup: coming soon.")
    console.print()
