"""WebUI setup: Node check, npm install, .env configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

from desdeo.cli.checks import check_node, check_npm, check_nvm
from desdeo.cli.styles import console, fail, step_header, success, warn

webui_app = typer.Typer(help="Set up the DESDEO web UI.")

WEBUI_DIR = Path(__file__).resolve().parent.parent.parent / "webui"


def _check_node_version() -> bool:
    """Check Node.js version and offer nvm switch if needed."""
    node_check = check_node()
    npm_check = check_npm()
    nvm_check = check_nvm()

    if node_check.ok:
        success(f"Node.js: {node_check.version}")
    else:
        if node_check.version:
            warn(f"Node.js: {node_check.version} (>= 24 recommended)")
        else:
            fail("Node.js not found")
            console.print("    Install Node.js >= 24 (https://nodejs.org/)")
            return False

        if nvm_check.ok:
            console.print("    nvm is available. Run: [bold]nvm install 24 && nvm use 24[/bold]")
            use_nvm = typer.confirm("    Try to switch now?", default=True)
            if use_nvm:
                # Source nvm and switch — this only works in the subprocess
                nvm_dir = os.environ.get("NVM_DIR", str(Path("~/.nvm").expanduser()))
                result = subprocess.run(
                    f'source "{nvm_dir}/nvm.sh" && nvm use 24 && node --version',
                    shell=True,
                    capture_output=True,
                    text=True,
                    executable="/bin/bash",
                )
                if result.returncode == 0:
                    success(f"Switched to Node {result.stdout.strip()}")
                else:
                    warn("Could not switch. Please run 'nvm use 24' manually before continuing.")
                    return False

    if not npm_check.ok:
        fail("npm not found")
        return False

    return True


def _run_npm_install(webui_dir: Path) -> bool:
    """Run npm install in the webui directory."""
    console.print("\n  Running npm install...\n")

    # Use nvm if available
    nvm_dir = os.environ.get("NVM_DIR", str(Path("~/.nvm").expanduser()))
    nvm_script = Path(nvm_dir) / "nvm.sh"

    if nvm_script.exists():
        cmd = f'source "{nvm_script}" && nvm use 24 2>/dev/null; npm install'
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
