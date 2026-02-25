"""Unit tests for conda environment support in the DESDEO CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# is_conda_env()
# ---------------------------------------------------------------------------


class TestIsCondaEnv:
    """Tests for desdeo.cli.config.is_conda_env."""

    def test_true_when_conda_prefix_set(self, monkeypatch):
        monkeypatch.setenv("CONDA_PREFIX", "/home/user/miniconda3/envs/test")
        from desdeo.cli.config import is_conda_env

        assert is_conda_env() is True

    def test_false_when_conda_prefix_unset(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        from desdeo.cli.config import is_conda_env

        assert is_conda_env() is False

    def test_false_when_conda_prefix_empty(self, monkeypatch):
        monkeypatch.setenv("CONDA_PREFIX", "")
        from desdeo.cli.config import is_conda_env

        assert is_conda_env() is False


# ---------------------------------------------------------------------------
# check_node() — relaxed version in conda
# ---------------------------------------------------------------------------


class TestCheckNodeConda:
    """Tests for check_node() conda version relaxation."""

    def test_node_18_ok_in_conda(self, monkeypatch):
        """Node 18 should be accepted inside a conda env."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setattr(
            "desdeo.cli.checks._run_version",
            lambda cmd: "v18.20.4" if cmd == "node" else None,
        )
        from desdeo.cli.checks import check_node

        result = check_node()
        assert result.ok is True
        assert result.version == "v18.20.4"
        assert "conda" in result.detail

    def test_node_22_ok_in_conda(self, monkeypatch):
        """Node 22 (common on conda-forge) should be accepted."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setattr(
            "desdeo.cli.checks._run_version",
            lambda cmd: "v22.1.0" if cmd == "node" else None,
        )
        from desdeo.cli.checks import check_node

        result = check_node()
        assert result.ok is True
        assert result.version == "v22.1.0"

    def test_node_18_not_ok_outside_conda(self, monkeypatch):
        """Node 18 should NOT be accepted outside conda."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(
            "desdeo.cli.checks._run_version",
            lambda cmd: "v18.20.4" if cmd == "node" else None,
        )
        from desdeo.cli.checks import check_node

        result = check_node()
        assert result.ok is False

    def test_node_24_ok_everywhere(self, monkeypatch):
        """Node >= 24 should be accepted regardless of conda."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr(
            "desdeo.cli.checks._run_version",
            lambda cmd: "v24.1.0" if cmd == "node" else None,
        )
        from desdeo.cli.checks import check_node

        result = check_node()
        assert result.ok is True

    def test_node_missing_not_ok_in_conda(self, monkeypatch):
        """Missing node is still not ok, even in conda."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setattr(
            "desdeo.cli.checks._run_version",
            lambda cmd: None,
        )
        from desdeo.cli.checks import check_node

        result = check_node()
        assert result.ok is False


# ---------------------------------------------------------------------------
# _add_to_path() — conda activation script branch
# ---------------------------------------------------------------------------


class TestAddToPathConda:
    """Tests for the conda branch of solvers._add_to_path()."""

    def test_writes_sh_activation_script(self, tmp_path, monkeypatch):
        """On Unix + conda, writes a .sh activation script."""
        conda_prefix = tmp_path / "envs" / "myenv"
        conda_prefix.mkdir(parents=True)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
        monkeypatch.setattr("sys.platform", "linux")

        # Auto-confirm the typer.confirm prompt
        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", lambda *a, **kw: True)

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        script = conda_prefix / "etc" / "conda" / "activate.d" / "desdeo-solvers.sh"
        assert script.exists()
        content = script.read_text()
        assert str(solver_dir) in content
        assert "export PATH=" in content

    def test_writes_bat_activation_script_on_win32(self, tmp_path, monkeypatch):
        """On Windows + conda, writes a .bat activation script."""
        conda_prefix = tmp_path / "envs" / "myenv"
        conda_prefix.mkdir(parents=True)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
        monkeypatch.setattr("sys.platform", "win32")

        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", lambda *a, **kw: True)

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        script = conda_prefix / "etc" / "conda" / "activate.d" / "desdeo-solvers.bat"
        assert script.exists()
        content = script.read_text()
        assert str(solver_dir) in content
        assert "@set" in content

    def test_skips_if_script_already_has_path(self, tmp_path, monkeypatch):
        """If the activation script already contains the solver dir, skip."""
        conda_prefix = tmp_path / "envs" / "myenv"
        conda_prefix.mkdir(parents=True)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
        monkeypatch.setattr("sys.platform", "linux")

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        # Pre-create the activation script with the solver dir
        activate_d = conda_prefix / "etc" / "conda" / "activate.d"
        activate_d.mkdir(parents=True)
        script = activate_d / "desdeo-solvers.sh"
        script.write_text(f'export PATH="{solver_dir}:$PATH"\n')

        # confirm should NOT be called
        confirm_called = False

        def _mock_confirm(*a, **kw):
            nonlocal confirm_called
            confirm_called = True
            return True

        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", _mock_confirm)

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        assert not confirm_called

    def test_declined_conda_no_script_written(self, tmp_path, monkeypatch):
        """If user declines, no activation script is written."""
        conda_prefix = tmp_path / "envs" / "myenv"
        conda_prefix.mkdir(parents=True)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
        monkeypatch.setattr("sys.platform", "linux")

        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", lambda *a, **kw: False)

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        script = conda_prefix / "etc" / "conda" / "activate.d" / "desdeo-solvers.sh"
        assert not script.exists()

    def test_updates_process_path_immediately(self, tmp_path, monkeypatch):
        """Even before writing the activation script, the process PATH is updated."""
        conda_prefix = tmp_path / "envs" / "myenv"
        conda_prefix.mkdir(parents=True)
        monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", lambda *a, **kw: True)

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        # Ensure solver_dir is NOT already in PATH
        monkeypatch.setenv("PATH", "/usr/bin:/usr/local/bin")

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        assert str(solver_dir) in os.environ["PATH"]

    def test_non_conda_falls_through_to_shell_rc(self, tmp_path, monkeypatch):
        """Without CONDA_PREFIX, _add_to_path uses shell rc logic."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("SHELL", "/bin/bash")

        # Create a fake .bashrc
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        bashrc.write_text("# existing bashrc\n")
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        monkeypatch.setattr("desdeo.cli.solvers.typer.confirm", lambda *a, **kw: True)

        solver_dir = tmp_path / "solvers"
        solver_dir.mkdir()

        from desdeo.cli.solvers import _add_to_path

        _add_to_path(solver_dir)

        # Should have written to .bashrc, not conda activation script
        assert str(solver_dir) in bashrc.read_text()


# ---------------------------------------------------------------------------
# _check_node_version() — conda install branch
# ---------------------------------------------------------------------------


class TestCheckNodeVersionConda:
    """Tests for the conda branch of webui._check_node_version()."""

    def test_conda_node_present_below_24_accepted(self, monkeypatch):
        """In conda with Node < 24, it should succeed without prompting."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        # check_node returns ok=True due to conda relaxation
        mock_check = MagicMock(ok=True, version="v20.11.0")
        monkeypatch.setattr("desdeo.cli.webui.check_node", lambda: mock_check)
        mock_npm = MagicMock(ok=True, version="9.0.0")
        monkeypatch.setattr("desdeo.cli.webui.check_npm", lambda: mock_npm)

        from desdeo.cli.webui import _check_node_version

        assert _check_node_version() is True

    def test_conda_node_missing_offers_conda_install(self, monkeypatch):
        """In conda with no Node, prompts for conda install (not nvm)."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        mock_check = MagicMock(ok=False, version=None)
        monkeypatch.setattr("desdeo.cli.webui.check_node", lambda: mock_check)

        # User chooses "2" (skip)
        monkeypatch.setattr("desdeo.cli.webui.typer.prompt", lambda *a, **kw: "2")

        from desdeo.cli.webui import _check_node_version

        assert _check_node_version() is False

    def test_conda_node_missing_installs_via_conda(self, monkeypatch):
        """In conda with no Node, choice '1' calls _install_node_via_conda."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        mock_check = MagicMock(ok=False, version=None)
        monkeypatch.setattr("desdeo.cli.webui.check_node", lambda: mock_check)

        monkeypatch.setattr("desdeo.cli.webui.typer.prompt", lambda *a, **kw: "1")

        install_called = False

        def mock_install():
            nonlocal install_called
            install_called = True
            return True

        monkeypatch.setattr("desdeo.cli.webui._install_node_via_conda", mock_install)

        mock_npm = MagicMock(ok=True, version="9.0.0")
        monkeypatch.setattr("desdeo.cli.webui.check_npm", lambda: mock_npm)

        from desdeo.cli.webui import _check_node_version

        result = _check_node_version()
        assert install_called
        assert result is True

    def test_non_conda_node_missing_offers_nvm(self, monkeypatch):
        """Outside conda on Unix, nvm is offered (not conda install)."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr("desdeo.cli.webui.sys.platform", "linux")
        mock_check = MagicMock(ok=False, version=None)
        monkeypatch.setattr("desdeo.cli.webui.check_node", lambda: mock_check)

        # User chooses "2" (skip)
        monkeypatch.setattr("desdeo.cli.webui.typer.prompt", lambda *a, **kw: "2")

        from desdeo.cli.webui import _check_node_version

        assert _check_node_version() is False


# ---------------------------------------------------------------------------
# _install_node_via_conda()
# ---------------------------------------------------------------------------


class TestInstallNodeViaConda:
    """Tests for webui._install_node_via_conda()."""

    def test_success(self, monkeypatch):
        """Successful conda install + node found afterwards."""
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        monkeypatch.setattr("desdeo.cli.webui.subprocess.run", lambda *a, **kw: mock_result)
        monkeypatch.setattr("desdeo.cli.webui.shutil.which", lambda cmd: "/opt/conda/bin/node")

        from desdeo.cli.webui import _install_node_via_conda

        assert _install_node_via_conda() is True

    def test_conda_install_fails(self, monkeypatch):
        """conda install returns non-zero."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="error: package not found")
        monkeypatch.setattr("desdeo.cli.webui.subprocess.run", lambda *a, **kw: mock_result)

        from desdeo.cli.webui import _install_node_via_conda

        assert _install_node_via_conda() is False

    def test_node_not_found_after_install(self, monkeypatch):
        """conda install succeeds but node binary not on PATH."""
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        monkeypatch.setattr("desdeo.cli.webui.subprocess.run", lambda *a, **kw: mock_result)
        monkeypatch.setattr("desdeo.cli.webui.shutil.which", lambda cmd: None)

        from desdeo.cli.webui import _install_node_via_conda

        assert _install_node_via_conda() is False


# ---------------------------------------------------------------------------
# wizard: conda detection message + default install choice
# ---------------------------------------------------------------------------


class TestWizardCondaDetection:
    """Tests for conda detection in wizard.setup() and _configure_install_paths()."""

    def test_configure_defaults_to_project_local_in_conda(self, tmp_path, monkeypatch):
        """_configure_install_paths() defaults to choice '3' (project-local) in conda."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/test")

        config_path = tmp_path / ".desdeo" / "config.toml"
        monkeypatch.setattr("desdeo.cli.config.get_config_path", lambda: config_path)

        # Track what default was passed to typer.prompt
        prompt_defaults = []

        def mock_prompt(text, *args, **kwargs):
            prompt_defaults.append(kwargs.get("default"))
            default = kwargs.get("default", "1")
            return default

        monkeypatch.setattr("desdeo.cli.wizard.typer.prompt", mock_prompt)
        monkeypatch.setattr("desdeo.cli.wizard.typer.confirm", lambda *a, **kw: True)

        from desdeo.cli.wizard import _configure_install_paths

        _configure_install_paths()

        # First prompt is the install location choice — should default to "3"
        assert prompt_defaults[0] == "3"

    def test_configure_defaults_to_default_outside_conda(self, tmp_path, monkeypatch):
        """_configure_install_paths() defaults to choice '1' (default) outside conda."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)

        config_path = tmp_path / ".desdeo" / "config.toml"
        monkeypatch.setattr("desdeo.cli.config.get_config_path", lambda: config_path)

        prompt_defaults = []

        def mock_prompt(text, *args, **kwargs):
            prompt_defaults.append(kwargs.get("default"))
            default = kwargs.get("default", "1")
            return default

        monkeypatch.setattr("desdeo.cli.wizard.typer.prompt", mock_prompt)
        monkeypatch.setattr("desdeo.cli.wizard.typer.confirm", lambda *a, **kw: True)

        from desdeo.cli.wizard import _configure_install_paths

        _configure_install_paths()

        # First prompt should default to "1"
        assert prompt_defaults[0] == "1"
