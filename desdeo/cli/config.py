"""Configurable install directories for the DESDEO CLI.

Persists install mode and resolved paths in `.desdeo/config.toml` at the
project root so that scripts, subsequent wizard runs, and `run_fullstack.sh`
can all find the installed tools without re-prompting.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path


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
