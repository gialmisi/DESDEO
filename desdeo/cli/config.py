"""Configurable install directories for the DESDEO CLI.

Persists install mode and resolved paths in `.desdeo/config.toml` at the
project root so that scripts, subsequent wizard runs, and `run_fullstack.sh`
can all find the installed tools without re-prompting.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from enum import Enum
from pathlib import Path


def is_conda_env() -> bool:
    """Return True if running inside an activated conda environment."""
    return bool(os.environ.get("CONDA_PREFIX"))


# ---------------------------------------------------------------------------
# Dependency-group helpers
# ---------------------------------------------------------------------------


def _find_pyproject_toml() -> Path | None:
    """Locate pyproject.toml for reading dependency groups.

    Tries two locations:
    1. Project root (dev/editable installs, cloned repos)
    2. Bundled ``desdeo/_pyproject.toml`` (pip-installed from wheel)
    """
    # 1. Dev / editable install: project root has pyproject.toml
    project_pyproject = get_project_root() / "pyproject.toml"
    if project_pyproject.is_file():
        return project_pyproject

    # 2. Pip-installed wheel: force-included as desdeo/_pyproject.toml
    bundled = Path(__file__).resolve().parent.parent / "_pyproject.toml"
    if bundled.is_file():
        return bundled

    return None


def read_dependency_groups() -> dict[str, list[str]]:
    """Parse ``[dependency-groups]`` from pyproject.toml.

    Returns ``{group_name: [specifier_strings]}``, filtering out
    ``include-group`` dict entries (PEP 735 cross-references).
    Returns ``{}`` if pyproject.toml is not found.
    """
    path = _find_pyproject_toml()
    if path is None:
        return {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    groups_raw = data.get("dependency-groups", {})
    result: dict[str, list[str]] = {}
    for name, entries in groups_raw.items():
        # Keep only plain string specifiers, skip include-group dicts
        result[name] = [e for e in entries if isinstance(e, str)]
    return result


def check_dependency_group(group: str) -> tuple[list[str], list[str]]:
    """Check which packages in a dependency group are missing.

    Returns ``(missing_specifiers, installed_names)``.
    """
    groups = read_dependency_groups()
    specs = groups.get(group, [])

    missing: list[str] = []
    installed: list[str] = []

    for spec in specs:
        # Extract distribution name: strip extras, version constraints, env markers
        dist_name = re.split(r"[\[>=<~!;]", spec)[0].strip()
        try:
            importlib.metadata.distribution(dist_name)
            installed.append(dist_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)

    return missing, installed


def ensure_dependency_groups(groups: list[str]) -> dict[str, bool]:
    """Check requested dependency groups and pip-install any missing packages.

    Returns ``{group_name: is_available}`` where ``is_available`` is True when
    all packages in that group are installed (either already or after install).
    """
    all_missing: list[str] = []
    group_missing: dict[str, list[str]] = {}

    for group in groups:
        m, _ = check_dependency_group(group)
        group_missing[group] = m
        all_missing.extend(m)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_missing: list[str] = []
    for spec in all_missing:
        if spec not in seen:
            seen.add(spec)
            unique_missing.append(spec)

    if unique_missing:
        from desdeo.cli.styles import console, fail, success

        console.print("\n  [bold]Installing missing dependencies...[/bold]")
        for group in groups:
            if group_missing[group]:
                names = ", ".join(re.split(r"[\[>=<~!;]", s)[0].strip() for s in group_missing[group])
                console.print(f"    {group}: {names}")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *unique_missing],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            success("Dependencies installed successfully.")
        else:
            fail(f"pip install failed (exit {result.returncode}).")
            console.print(f"    [dim]{result.stderr.strip()[:200]}[/dim]")

    # Re-check after install
    status: dict[str, bool] = {}
    for group in groups:
        m, _ = check_dependency_group(group)
        status[group] = len(m) == 0

    return status


class InstallMode(Enum):
    """Where CLI-managed tools are installed."""

    DEFAULT = "default"
    CUSTOM = "custom"
    PROJECT_LOCAL = "project_local"


def get_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: two levels up from desdeo/cli/
    return Path(__file__).resolve().parent.parent.parent


def get_config_dir() -> Path:
    """Return the `.desdeo/` directory at the project root."""
    return get_project_root() / ".desdeo"


def get_config_path() -> Path:
    """Return the path to `.desdeo/config.toml`."""
    return get_config_dir() / "config.toml"


def config_exists() -> bool:
    """Return True if a config file already exists."""
    return get_config_path().is_file()


def _default_solver_dir() -> str:
    """Platform-appropriate default solver directory."""
    if sys.platform == "win32":
        local = Path(Path.home(), "AppData", "Local", "desdeo", "solvers")
        return str(local)
    return str(Path.home() / ".local" / "share" / "desdeo" / "solvers")


def _default_nvm_dir() -> str:
    """Platform-appropriate default nvm directory."""
    return str(Path.home() / ".nvm")


def resolve_paths_for_mode(
    mode: InstallMode,
    custom_base: str | None = None,
) -> dict[str, str]:
    """Compute absolute paths for a given install mode.

    Returns a dict with keys ``solvers`` and ``nvm_dir``.
    """
    if mode is InstallMode.PROJECT_LOCAL:
        root = get_project_root()
        return {
            "solvers": str(root / ".desdeo" / "solvers"),
            "nvm_dir": str(root / ".desdeo" / "nvm"),
        }
    if mode is InstallMode.CUSTOM:
        if not custom_base:
            raise ValueError("custom_base is required for CUSTOM mode")
        base = Path(custom_base).expanduser().resolve()
        return {
            "solvers": str(base / "solvers"),
            "nvm_dir": str(base / "nvm"),
        }
    # DEFAULT
    return {
        "solvers": _default_solver_dir(),
        "nvm_dir": _default_nvm_dir(),
    }


def save_config(
    mode: InstallMode,
    solvers: str,
    nvm_dir: str,
    custom_base: str | None = None,
) -> Path:
    """Write config to `.desdeo/config.toml` and return the path.

    Uses plain string formatting to avoid a ``tomli_w`` dependency.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize backslashes for TOML (forward slashes work everywhere)
    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    lines = [
        "[install]",
        f'mode = "{mode.value}"',
    ]
    if custom_base:
        lines.append(f'custom_base = "{_norm(custom_base)}"')

    lines += [
        "",
        "[paths]",
        f'solvers = "{_norm(solvers)}"',
        f'nvm_dir = "{_norm(nvm_dir)}"',
    ]

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def load_config() -> dict:
    """Load config from `.desdeo/config.toml`.

    Returns a dict like::

        {
            "mode": "default",
            "custom_base": None,
            "solvers": "/home/user/.local/share/desdeo/solvers",
            "nvm_dir": "/home/user/.nvm",
        }

    If the config file does not exist, returns defaults.
    """
    path = get_config_path()
    if not path.is_file():
        defaults = resolve_paths_for_mode(InstallMode.DEFAULT)
        return {
            "mode": InstallMode.DEFAULT.value,
            "custom_base": None,
            **defaults,
        }

    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    install = data.get("install", {})
    paths = data.get("paths", {})
    return {
        "mode": install.get("mode", "default"),
        "custom_base": install.get("custom_base"),
        "solvers": paths.get("solvers", _default_solver_dir()),
        "nvm_dir": paths.get("nvm_dir", _default_nvm_dir()),
    }


def get_solver_dir() -> str:
    """Convenience: return the configured solver directory."""
    return load_config()["solvers"]


def get_nvm_dir() -> str:
    """Convenience: return the configured nvm directory."""
    return load_config()["nvm_dir"]
