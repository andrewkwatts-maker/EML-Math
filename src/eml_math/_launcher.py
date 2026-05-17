"""Optional companion-app launcher.

Searches for the EML-Math-App directory as a sibling of the library checkout
(developer path). Falls back to ~/.eml-math-app, cloning from GitHub at the
matching version tag if neither exists.
"""
import os
import subprocess
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

_APP_REPO   = "https://github.com/andrewkwatts-maker/EML-Math-App.git"
_APP_FOLDER = "EML-Math-App"
_APP_MODULE = "eml_math_app"
_LIB_FILE   = Path(__file__)


def _library_version() -> str:
    try:
        return _pkg_version("eml-math")
    except Exception:
        return ""


def _find_or_clone() -> Path:
    for parent in _LIB_FILE.parents:
        candidate = parent / _APP_FOLDER
        if candidate.is_dir():
            return candidate

    home_dir = Path.home() / ".eml-math-app"
    if home_dir.is_dir():
        return home_dir

    tag = f"v{_library_version()}"
    print(f"Cloning {_APP_FOLDER} @ {tag} into {home_dir} ...")
    result = subprocess.run(["git", "clone", "--branch", tag, _APP_REPO, str(home_dir)])
    if result.returncode != 0:
        print(f"Tag {tag} not found on app repo — cloning default branch ...")
        if subprocess.run(["git", "clone", _APP_REPO, str(home_dir)]).returncode != 0:
            sys.exit("git clone failed — is git installed?")

    print("Installing app dependencies ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(home_dir)], check=True)
    return home_dir


def launch():
    app_dir = _find_or_clone()
    os.chdir(str(app_dir))
    subprocess.run([sys.executable, "-m", _APP_MODULE] + sys.argv[1:])
