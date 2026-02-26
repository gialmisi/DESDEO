"""Unit tests for uv-based dependency-group helpers in desdeo.cli.config."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from desdeo.cli.config import ensure_uv, get_project_root, uv_sync_groups


# ---------------------------------------------------------------------------
# ensure_uv()
# ---------------------------------------------------------------------------


class TestEnsureUv:
    def test_returns_true_when_uv_on_path(self, monkeypatch):
        monkeypatch.setattr("desdeo.cli.config.shutil.which", lambda cmd: "/usr/bin/uv")
        assert ensure_uv() is True

    def test_returns_false_when_missing_outside_conda(self, monkeypatch):
        monkeypatch.setattr("desdeo.cli.config.shutil.which", lambda cmd: None)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        assert ensure_uv() is False

    def test_offers_conda_install_when_missing_in_conda(self, monkeypatch):
        monkeypatch.setattr("desdeo.cli.config.shutil.which", lambda cmd: None)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/test")
        monkeypatch.setattr("desdeo.cli.config.typer.confirm", lambda *a, **kw: True)

        mock_result = MagicMock(returncode=0, stderr="")
        call_args = []

        def mock_run(cmd, **kwargs):
            call_args.append(cmd)
            return mock_result

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        assert ensure_uv() is True
        assert call_args[0] == ["conda", "install", "-y", "conda-forge::uv"]

    def test_conda_install_declined(self, monkeypatch):
        monkeypatch.setattr("desdeo.cli.config.shutil.which", lambda cmd: None)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/test")
        monkeypatch.setattr("desdeo.cli.config.typer.confirm", lambda *a, **kw: False)

        assert ensure_uv() is False

    def test_conda_install_fails(self, monkeypatch):
        monkeypatch.setattr("desdeo.cli.config.shutil.which", lambda cmd: None)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/test")
        monkeypatch.setattr("desdeo.cli.config.typer.confirm", lambda *a, **kw: True)

        mock_result = MagicMock(returncode=1, stderr="error")
        monkeypatch.setattr("desdeo.cli.config.subprocess.run", lambda *a, **kw: mock_result)

        assert ensure_uv() is False


# ---------------------------------------------------------------------------
# uv_sync_groups()
# ---------------------------------------------------------------------------


class TestUvSyncGroups:
    def test_calls_uv_sync_with_group_flags(self, monkeypatch):
        call_args = []

        def mock_run(cmd, **kwargs):
            call_args.append((cmd, kwargs))
            return MagicMock(returncode=0)

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        result = uv_sync_groups(["web", "dev"])

        assert result is True
        cmd = call_args[0][0]
        assert cmd == ["uv", "sync", "--group", "web", "--group", "dev"]
        assert call_args[0][1]["cwd"] == str(get_project_root())

    def test_returns_false_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            "desdeo.cli.config.subprocess.run",
            lambda *a, **kw: MagicMock(returncode=1),
        )

        assert uv_sync_groups(["web"]) is False

    def test_single_group(self, monkeypatch):
        call_args = []

        def mock_run(cmd, **kwargs):
            call_args.append(cmd)
            return MagicMock(returncode=0)

        monkeypatch.setattr("desdeo.cli.config.subprocess.run", mock_run)

        uv_sync_groups(["docs"])
        assert call_args[0] == ["uv", "sync", "--group", "docs"]
