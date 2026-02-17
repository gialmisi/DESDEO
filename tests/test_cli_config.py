"""Unit tests for desdeo.cli.config — install directory configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from desdeo.cli.config import (
    InstallMode,
    get_project_root,
    load_config,
    resolve_paths_for_mode,
    save_config,
)


def test_get_project_root():
    """get_project_root() finds the repo root containing pyproject.toml."""
    root = get_project_root()
    assert (root / "pyproject.toml").exists()


def test_default_config_paths():
    """resolve_paths_for_mode(DEFAULT) returns platform-appropriate paths."""
    paths = resolve_paths_for_mode(InstallMode.DEFAULT)
    assert "solvers" in paths
    assert "nvm_dir" in paths
    # Paths should be absolute
    assert Path(paths["solvers"]).is_absolute()
    assert Path(paths["nvm_dir"]).is_absolute()


def test_resolve_project_local():
    """resolve_paths_for_mode(PROJECT_LOCAL) puts paths under .desdeo/."""
    paths = resolve_paths_for_mode(InstallMode.PROJECT_LOCAL)
    root = get_project_root()
    assert paths["solvers"] == str(root / ".desdeo" / "solvers")
    assert paths["nvm_dir"] == str(root / ".desdeo" / "nvm")


def test_resolve_custom():
    """resolve_paths_for_mode(CUSTOM) puts paths under the custom base."""
    paths = resolve_paths_for_mode(InstallMode.CUSTOM, "/opt/desdeo")
    assert paths["solvers"] == str(Path("/opt/desdeo/solvers").resolve())
    assert paths["nvm_dir"] == str(Path("/opt/desdeo/nvm").resolve())


def test_resolve_custom_requires_base():
    """CUSTOM mode without custom_base raises ValueError."""
    with pytest.raises(ValueError, match="custom_base"):
        resolve_paths_for_mode(InstallMode.CUSTOM)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """save_config() then load_config() returns the same values."""
    config_path = tmp_path / ".desdeo" / "config.toml"

    # Patch get_config_path to use tmp_path
    monkeypatch.setattr(
        "desdeo.cli.config.get_config_path",
        lambda: config_path,
    )

    solver_dir = str(tmp_path / "my_solvers")
    nvm_dir = str(tmp_path / "my_nvm")

    save_config(
        mode=InstallMode.CUSTOM,
        solvers=solver_dir,
        nvm_dir=nvm_dir,
        custom_base=str(tmp_path),
    )

    assert config_path.is_file()
    cfg = load_config()
    assert cfg["mode"] == "custom"
    # Backslashes are normalized to forward slashes in the TOML file
    assert cfg["solvers"] == solver_dir.replace("\\", "/")
    assert cfg["nvm_dir"] == nvm_dir.replace("\\", "/")
    assert cfg["custom_base"] == str(tmp_path).replace("\\", "/")


def test_config_not_exists_returns_defaults(tmp_path, monkeypatch):
    """load_config() returns defaults when no file exists."""
    monkeypatch.setattr(
        "desdeo.cli.config.get_config_path",
        lambda: tmp_path / "nonexistent" / "config.toml",
    )

    cfg = load_config()
    assert cfg["mode"] == "default"
    assert cfg["custom_base"] is None
    assert Path(cfg["solvers"]).is_absolute()
    assert Path(cfg["nvm_dir"]).is_absolute()


def test_save_creates_parent_dirs(tmp_path, monkeypatch):
    """save_config() creates the .desdeo/ directory if it doesn't exist."""
    config_path = tmp_path / "deep" / "nested" / "config.toml"
    monkeypatch.setattr(
        "desdeo.cli.config.get_config_path",
        lambda: config_path,
    )

    save_config(
        mode=InstallMode.DEFAULT,
        solvers="/some/solvers",
        nvm_dir="/some/nvm",
    )
    assert config_path.is_file()


def test_windows_paths_in_config(tmp_path, monkeypatch):
    """Backslashes in Windows paths are normalized to forward slashes."""
    config_path = tmp_path / ".desdeo" / "config.toml"
    monkeypatch.setattr(
        "desdeo.cli.config.get_config_path",
        lambda: config_path,
    )

    win_solver = "C:\\Users\\alice\\AppData\\Local\\desdeo\\solvers"
    win_nvm = "C:\\Users\\alice\\.nvm"

    save_config(
        mode=InstallMode.DEFAULT,
        solvers=win_solver,
        nvm_dir=win_nvm,
    )

    cfg = load_config()
    # Backslashes should have been normalized to forward slashes
    assert "\\" not in cfg["solvers"]
    assert "\\" not in cfg["nvm_dir"]
    assert cfg["solvers"] == "C:/Users/alice/AppData/Local/desdeo/solvers"
    assert cfg["nvm_dir"] == "C:/Users/alice/.nvm"


def test_project_local_mode_roundtrip(tmp_path, monkeypatch):
    """PROJECT_LOCAL mode saves and loads correctly."""
    config_path = tmp_path / ".desdeo" / "config.toml"
    monkeypatch.setattr(
        "desdeo.cli.config.get_config_path",
        lambda: config_path,
    )

    paths = resolve_paths_for_mode(InstallMode.PROJECT_LOCAL)
    save_config(
        mode=InstallMode.PROJECT_LOCAL,
        solvers=paths["solvers"],
        nvm_dir=paths["nvm_dir"],
    )

    cfg = load_config()
    assert cfg["mode"] == "project_local"
