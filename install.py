import json
import os
import subprocess
import sys
from pathlib import Path


SUPPORTED_PLATFORMS = {"win32", "linux"}


def check_environment() -> bool:
    """Validate python version, platform and git installation."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print("ERROR: Python >= 3.10 required")
        return False
    if sys.platform not in SUPPORTED_PLATFORMS:
        print(f"ERROR: unsupported platform {sys.platform}")
        return False
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=sys.platform == "linux",
        )
    except Exception:
        print("ERROR: git is not installed, please install git")
        return False
    return True


def create_virtualenv() -> Path:
    """Create venv and install python requirements."""
    python = sys.executable
    subprocess.check_call(f"{python} -m venv venv", shell=sys.platform == "linux")
    pip = Path("venv/Scripts/pip.exe" if sys.platform == "win32" else "venv/bin/pip")
    subprocess.check_call(f"{pip} install -U -r requirements.txt", shell=sys.platform == "linux")
    return pip


def configure_local_run() -> bool:
    """Ask user if backend should run locally and update config."""
    config = Path("config.json")
    data = json.loads(config.read_text()) if config.exists() else {}

    choice = None
    while choice not in ("y", "n"):
        choice = input("Are you using this locally? (y/n): ").lower()

    run_local = choice == "y"
    data["run_local"] = run_local
    config.write_text(json.dumps(data, indent=2))
    return run_local


def update_submodules() -> None:
    subprocess.check_call("git submodule update --init --recursive", shell=sys.platform == "linux")


def run_backend_installer(python: Path) -> None:
    """Execute the backend installer inside its directory."""
    os.chdir("backend")
    try:
        subprocess.check_call(f"{python} installer.py local", shell=sys.platform == "linux")
    finally:
        os.chdir("..")


def main() -> None:
    if not check_environment():
        return

    create_virtualenv()
    if configure_local_run():
        update_submodules()
        python = Path("venv/Scripts/python.exe" if sys.platform == "win32" else "venv/bin/python")
        if Path("backend/sd_scripts").exists():
            run_backend_installer(python)
        else:
            print("ERROR: backend submodules not found after update.")


if __name__ == "__main__":
    main()
