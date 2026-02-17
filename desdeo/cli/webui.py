"""WebUI setup: Node check, npm install, .env configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from desdeo.cli.checks import check_node, check_npm
from desdeo.cli.styles import console, fail, info, step_header, success, warn

webui_app = typer.Typer(help="Set up the DESDEO web UI.")

WEBUI_DIR = Path(__file__).resolve().parent.parent.parent / "webui"


def _get_nvm_dir() -> str:
    """Return the nvm directory from config, env, or default."""
    from desdeo.cli.config import config_exists, get_nvm_dir

    if config_exists():
        return get_nvm_dir()
    return os.environ.get("NVM_DIR", str(Path("~/.nvm").expanduser()))


def _install_node_via_nvm() -> bool:
    """Install Node.js 24 via nvm, adding the binary to PATH for this process."""
    nvm_dir = _get_nvm_dir()
    nvm_script = Path(nvm_dir) / "nvm.sh"

    # Install nvm if not present
    if not nvm_script.exists():
        console.print("  Installing nvm...")
        install_env = os.environ.copy()
        install_env["NVM_DIR"] = nvm_dir
        result = subprocess.run(
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
            env=install_env,
        )
        if result.returncode != 0:
            fail("nvm installation failed")
            console.print(f"    {result.stderr.strip()}")
            return False
        # Update nvm_dir after installation
        nvm_dir = _get_nvm_dir()
        nvm_script = Path(nvm_dir) / "nvm.sh"
        if not nvm_script.exists():
            fail("nvm.sh not found after installation")
            return False
        success("nvm installed")

    # Install Node 24 and capture the node binary path
    console.print("  Installing Node.js 24 via nvm...")
    result = subprocess.run(
        f'source "{nvm_script}" && nvm install 24 && which node',
        shell=True,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )
    if result.returncode != 0:
        fail("Node.js 24 installation failed")
        console.print(f"    {result.stderr.strip()}")
        return False

    # Extract node binary directory and add to PATH for this process
    lines = result.stdout.strip().splitlines()
    node_path = lines[-1] if lines else ""
    if node_path and os.path.isfile(node_path):
        node_bin_dir = str(Path(node_path).parent)
        os.environ["PATH"] = node_bin_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ["NVM_DIR"] = nvm_dir
        success(f"Node.js installed ({node_bin_dir})")
        return True

    fail("Could not determine node binary path after installation")
    return False


def _check_node_version() -> bool:
    """Check Node.js version and offer nvm install if needed."""
    node_check = check_node()

    if node_check.ok:
        success(f"Node.js: {node_check.version}")
    else:
        if node_check.version:
            warn(f"Node.js: {node_check.version} (>= 24 required)")
        else:
            fail("Node.js not found")

        if sys.platform == "win32":
            console.print("    Install Node.js >= 24 from https://nodejs.org/")
            return False

        console.print("\n  Options:")
        console.print("    1) Install Node.js 24 via nvm (recommended)")
        console.print("    2) Skip (install manually)\n")

        choice = typer.prompt("  Choice", default="1")

        if choice == "1":
            if not _install_node_via_nvm():
                return False
        else:
            info("Skipping Node.js installation.")
            console.print("    Install Node.js >= 24 and re-run this command.")
            return False

    npm_check = check_npm()
    if not npm_check.ok:
        fail("npm not found")
        return False

    return True


def _run_npm_install(webui_dir: Path) -> bool:
    """Run npm install in the webui directory."""
    console.print("\n  Running npm install...\n")

    # Use nvm if available
    nvm_dir = _get_nvm_dir()
    nvm_script = Path(nvm_dir) / "nvm.sh"

    if nvm_script.exists():
        cmd = f'source "{nvm_script}" && npm install'
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=webui_dir,
            executable="/bin/bash",
        )
    else:
        result = subprocess.run(
            ["npm", "install"],
            cwd=webui_dir,
        )

    if result.returncode == 0:
        success("npm install completed")
        return True
    fail("npm install failed")
    return False


def _setup_env(webui_dir: Path) -> None:
    """Create or update .env file for the webui."""
    env_file = webui_dir / ".env"
    current_api_base = "http://localhost:8000"
    current_vite_api = "/api"

    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("API_BASE_URL="):
                current_api_base = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("VITE_API_URL="):
                current_vite_api = line.split("=", 1)[1].strip().strip('"')

    api_base = typer.prompt("  API backend URL", default=current_api_base)
    vite_api = typer.prompt("  VITE_API_URL", default=current_vite_api)

    env_content = f'API_BASE_URL="{api_base}"\nVITE_API_URL="{vite_api}"\n'
    env_file.write_text(env_content)
    success(f".env written to {env_file}")


@webui_app.callback(invoke_without_command=True)
def webui() -> None:
    """Set up the DESDEO web UI."""
    webui_dir = WEBUI_DIR
    if not webui_dir.exists():
        fail(f"WebUI directory not found at {webui_dir}")
        return

    step_header(1, 3, "Node.js Check")
    if not _check_node_version():
        return

    step_header(2, 3, "Install Dependencies")
    if not _run_npm_install(webui_dir):
        return

    step_header(3, 3, "Environment Configuration")
    _setup_env(webui_dir)

    console.print()
    success("WebUI setup complete!")
    console.print()
