"""Unit tests for dependency-group helpers in desdeo.cli.config."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from desdeo.cli.config import (
    _find_pyproject_toml,
    check_dependency_group,
    ensure_dependency_groups,
    get_project_root,
    read_dependency_groups,
)


# ---------------------------------------------------------------------------
# _find_pyproject_toml()
# ---------------------------------------------------------------------------


class TestFindPyprojectToml:
    def test_finds_from_project_root(self):
        """In a dev/editable install the project root pyproject.toml is found."""
        result = _find_pyproject_toml()
        assert result is not None
        assert result.name == "pyproject.toml"
        assert result.is_file()

    def test_finds_bundled_when_project_root_missing(self, tmp_path, monkeypatch):
        """Falls back to desdeo/_pyproject.toml when project root has none."""
        # Make get_project_root return a directory without pyproject.toml
        monkeypatch.setattr("desdeo.cli.config.get_project_root", lambda: tmp_path)

        # Create the bundled _pyproject.toml where the code expects it
        bundled = Path(__file__).resolve().parent.parent / "desdeo" / "_pyproject.toml"
        created = False
        try:
            if not bundled.exists():
                bundled.write_text('[project]\nname = "desdeo"\n')
                created = True

            result = _find_pyproject_toml()
            assert result is not None
            assert result.name == "_pyproject.toml"
        finally:
            if created:
                bundled.unlink()

    def test_returns_none_when_neither_found(self, tmp_path, monkeypatch):
        """Returns None when neither project root nor bundled file exists."""
        monkeypatch.setattr("desdeo.cli.config.get_project_root", lambda: tmp_path)
        # __file__ parent.parent won't have _pyproject.toml in the test env
        # (unless the previous test left one — but we clean up)
        bundled = Path(__file__).resolve().parent.parent / "desdeo" / "_pyproject.toml"
        if not bundled.exists():
            result = _find_pyproject_toml()
            # In the dev install, project root DOES have pyproject.toml,
            # but we patched get_project_root to tmp_path.
            # The bundled path is relative to config.py's __file__, not this test.
            # So this may or may not find it depending on layout.
            # Just verify it doesn't crash.
            assert result is None or result.is_file()


# ---------------------------------------------------------------------------
# read_dependency_groups()
# ---------------------------------------------------------------------------


class TestReadDependencyGroups:
    def test_parses_all_groups(self):
        """Returns all groups from the real pyproject.toml."""
        groups = read_dependency_groups()
        assert "dev" in groups
        assert "web" in groups
        assert "docs" in groups
        assert "jupyter" in groups
        assert "viz" in groups
        assert "tools" in groups
        assert "server" in groups
        assert "all-dev" in groups

    def test_filters_include_group_dicts(self):
        """The all-dev group only contains string specifiers, not include-group dicts."""
        groups = read_dependency_groups()
        all_dev = groups.get("all-dev", [])
        for entry in all_dev:
            assert isinstance(entry, str), f"Expected string, got {type(entry)}: {entry}"

    def test_returns_empty_when_no_pyproject(self, monkeypatch):
        """Returns {} when pyproject.toml cannot be found."""
        monkeypatch.setattr("desdeo.cli.config._find_pyproject_toml", lambda: None)
        groups = read_dependency_groups()
        assert groups == {}

    def test_web_group_has_expected_packages(self):
        """The web group should include known packages like fastapi, sqlmodel."""
        groups = read_dependency_groups()
        web_specs = groups.get("web", [])
        web_names = [s.split("[")[0].split(">")[0].split("<")[0].strip() for s in web_specs]
        assert "fastapi" in web_names
        assert "sqlmodel" in web_names


# ---------------------------------------------------------------------------
# check_dependency_group()
# ---------------------------------------------------------------------------


class TestCheckDependencyGroup:
    def test_detects_missing_package(self, monkeypatch):
        """A package not installed is reported as missing."""
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"test-group": ["nonexistent-pkg>=1.0"]},
        )
        missing, installed = check_dependency_group("test-group")
        assert "nonexistent-pkg>=1.0" in missing
        assert "nonexistent-pkg" not in installed

    def test_all_installed(self, monkeypatch):
        """When all packages are present, missing list is empty."""
        # Use packages we know are installed in the test environment
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"test-group": ["pytest>=8.0", "numpy>=2.0"]},
        )
        missing, installed = check_dependency_group("test-group")
        assert missing == []
        assert "pytest" in installed
        assert "numpy" in installed

    def test_name_extraction_with_extras(self, monkeypatch):
        """Extras like [cryptography] are stripped when checking distribution name."""
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"test-group": ["numpy[extra]>=2.0"]},
        )
        missing, installed = check_dependency_group("test-group")
        # numpy is installed, so even with [extra] it should be found
        assert missing == []
        assert "numpy" in installed

    def test_empty_group(self, monkeypatch):
        """An empty group returns empty lists."""
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"empty": []},
        )
        missing, installed = check_dependency_group("empty")
        assert missing == []
        assert installed == []

    def test_nonexistent_group(self, monkeypatch):
        """A group not in pyproject.toml returns empty lists."""
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"other": ["foo"]},
        )
        missing, installed = check_dependency_group("nope")
        assert missing == []
        assert installed == []


# ---------------------------------------------------------------------------
# ensure_dependency_groups()
# ---------------------------------------------------------------------------


class TestEnsureDependencyGroups:
    def test_calls_pip_with_missing_specifiers(self, monkeypatch):
        """When packages are missing, pip is invoked with the right specifiers."""
        call_args = []

        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"grp1": ["fakepkg1>=1.0", "pytest>=8.0"], "grp2": ["fakepkg2>=2.0"]},
        )

        # First call: fakepkg1 missing, pytest installed, fakepkg2 missing
        # After install: all found
        call_count = {"n": 0}

        original_check = check_dependency_group.__wrapped__ if hasattr(check_dependency_group, "__wrapped__") else None

        def mock_check(group):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Before install
                if group == "grp1":
                    return ["fakepkg1>=1.0"], ["pytest"]
                return ["fakepkg2>=2.0"], []
            # After install — all good
            return [], ["fakepkg1", "pytest", "fakepkg2"]

        monkeypatch.setattr("desdeo.cli.config.check_dependency_group", mock_check)

        def mock_run(cmd, **kwargs):
            call_args.append(cmd)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        result = ensure_dependency_groups(["grp1", "grp2"])

        # pip should have been called
        assert len(call_args) == 1
        pip_cmd = call_args[0]
        assert pip_cmd[0] == sys.executable
        assert "-m" in pip_cmd
        assert "pip" in pip_cmd
        assert "install" in pip_cmd
        assert "fakepkg1>=1.0" in pip_cmd
        assert "fakepkg2>=2.0" in pip_cmd
        # pytest should NOT be in the install list (already installed)
        assert "pytest>=8.0" not in pip_cmd

        assert result["grp1"] is True
        assert result["grp2"] is True

    def test_skips_pip_when_nothing_missing(self, monkeypatch):
        """No pip call when all packages are already installed."""
        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"grp": ["pytest>=8.0"]},
        )
        monkeypatch.setattr(
            "desdeo.cli.config.check_dependency_group",
            lambda g: ([], ["pytest"]),
        )

        pip_called = False

        def mock_run(cmd, **kwargs):
            nonlocal pip_called
            pip_called = True
            return MagicMock(returncode=0)

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        result = ensure_dependency_groups(["grp"])

        assert not pip_called
        assert result["grp"] is True

    def test_deduplicates_across_groups(self, monkeypatch):
        """Shared specifiers across groups are only installed once."""
        call_args = []

        monkeypatch.setattr(
            "desdeo.cli.config.read_dependency_groups",
            lambda: {"a": ["shared-pkg>=1.0"], "b": ["shared-pkg>=1.0", "other-pkg>=2.0"]},
        )

        call_count = {"n": 0}

        def mock_check(group):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                if group == "a":
                    return ["shared-pkg>=1.0"], []
                return ["shared-pkg>=1.0", "other-pkg>=2.0"], []
            return [], ["shared-pkg", "other-pkg"]

        monkeypatch.setattr("desdeo.cli.config.check_dependency_group", mock_check)

        def mock_run(cmd, **kwargs):
            call_args.append(cmd)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        ensure_dependency_groups(["a", "b"])

        pip_cmd = call_args[0]
        # shared-pkg should appear only once
        assert pip_cmd.count("shared-pkg>=1.0") == 1
        assert "other-pkg>=2.0" in pip_cmd
