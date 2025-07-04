import argparse
import json
import subprocess
import sys
from pathlib import Path

SUPPORTED_PLATFORMS = {"win32", "linux"}


def check_environment() -> bool:
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


def create_or_update_venv() -> Path:
    python = sys.executable
    subprocess.check_call(f"{python} -m venv venv", shell=sys.platform == "linux")
    pip = Path("venv/Scripts/pip.exe" if sys.platform == "win32" else "venv/bin/pip")
    subprocess.check_call(f"{pip} install -U -r requirements.txt", shell=sys.platform == "linux")
    return pip


def install_backend(python: str) -> None:
    subprocess.check_call("git submodule update --init --recursive", shell=sys.platform == "linux")
    backend_installer = Path("backend/installer.py")
    if backend_installer.exists():
        subprocess.check_call(f"{python} {backend_installer} local", shell=sys.platform == "linux")


def install() -> None:
    if not check_environment():
        return
    create_or_update_venv()
    config = Path("config.json")
    config_dict = json.loads(config.read_text()) if config.exists() else {}
    choice = None
    while choice not in ("y", "n"):
        choice = input("Are you using this locally? (y/n): ").lower()
    config_dict["run_local"] = choice == "y"
    config.write_text(json.dumps(config_dict, indent=2))
    if choice == "y":
        python = Path("venv/Scripts/python.exe" if sys.platform == "win32" else "venv/bin/python")
        install_backend(str(python))


def update() -> None:
    if not check_environment():
        return
    subprocess.check_call("git pull", shell=sys.platform == "linux")
    subprocess.check_call("git submodule update --init --recursive", shell=sys.platform == "linux")
    pip = Path("venv/Scripts/pip.exe" if sys.platform == "win32" else "venv/bin/pip")
    if pip.exists():
        subprocess.check_call(f"{pip} install -U -r requirements.txt", shell=sys.platform == "linux")
    else:
        print("Virtual environment not found. Run install first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installer for the Gradio LoRA Trainer")
    parser.add_argument("command", nargs="?", default="install", choices=["install", "update"], help="Action to perform")
    args = parser.parse_args()
    if args.command == "install":
        install()
    else:
        update()
