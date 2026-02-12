#!/usr/bin/env python3
"""Bootstrap: ensures uv is available, syncs deps, launches the real CLI.

This script uses only the standard library so it can run before `uv sync`.
Usage:
    python setup.py
"""

import os
import shutil
import subprocess
import sys


def main() -> int:
    """Bootstrap DESDEO setup."""
    # 1. Check for uv
    if not shutil.which("uv"):
        print("Error: 'uv' is not installed.")
        print()
        print("Install uv with:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print()
        print("Then run this script again:")
        print("  python setup.py")
        return 1

    # 2. Sync dependencies
    print("Syncing dependencies with uv...")
    result = subprocess.run(["uv", "sync"])
    if result.returncode != 0:
        print("Error: 'uv sync' failed.")
        return 1

    # 3. Launch the real CLI wizard
    os.execvp("uv", ["uv", "run", "python", "-m", "desdeo.cli", "setup"])  # noqa: S606

    # execvp replaces the process, so this is unreachable
    return 0


if __name__ == "__main__":
    sys.exit(main())
