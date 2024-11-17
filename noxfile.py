import os
import sys
from pathlib import Path

import nox

PACKAGE = "glmnet"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
LATEST_VERSION = PYTHON_VERSIONS[-1]
os.environ["PDM_IGNORE_SAVED_PYTHON"] = "1"
os.environ["PDM_IGNORE_ACTIVE_VENV"] = "0"
nox.needs_version = ">=2024.4.15"
nox.options.sessions = (
    "mypy",
    "tests",
)

locations = (
    "src",
    "tests",
)


@nox.session(python=LATEST_VERSION, reuse_venv=True)
def lockfile(session) -> None:
    """Run the test suite."""
    session.run_always("pdm", "lock", external=True)


@nox.session(python=LATEST_VERSION)
def lint(session) -> None:
    """Lint using ruff."""
    args = session.posargs or locations
    session.run("uv", "pip", "install", "ruff")
    session.run("ruff", "check", "--fix", *args)
    session.run("ruff", "format", *args)


@nox.session(python=LATEST_VERSION, reuse_venv=True)
def mypy(session) -> None:
    """Type-check using mypy."""
    session.run("pdm", "install", "--no-self", "--no-default", "--dev", external=True)
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        f"--python-executable={sys.executable}",
        "noxfile.py",
        external=True,
    )


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True) #PYTHON_VERSIONS
def tests(session) -> None:
    """Run the test suite."""
    session.run("uv", "pip", "install", "meson-python", "ninja", "setuptools", "numpy", "coverage[toml]", "pytest")
    session.run_always("pdm", "install", "--fail-fast", "--no-editable", "--frozen-lockfile", "--with", "dev", external=True)
    session.run(
        # running in parallel doesn't work I think because of not setting a seed
        # "coverage", "run", "--parallel", "-m", "pytest", "--numprocesses", "auto", "--random-order", external=True
        "coverage", "run", "-m", "pytest", external=True
    )


@nox.session(python=LATEST_VERSION, reuse_venv=True)
def coverage(session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]
    session.run("uv", "pip", "install", "coverage[toml]", "codecov", external=True)

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", "json", "--fail-under=0")
    session.run("codecov", *args)
