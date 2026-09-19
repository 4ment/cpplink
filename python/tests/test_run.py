# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""`cpplink.run` is the command line, in process."""

from __future__ import annotations

import subprocess
import sys

import cpplink


def test_version_matches_the_command_line() -> None:
    result = cpplink.run(["--version"])
    assert result.code == 0
    assert result.stdout.strip() == cpplink.__version__
    assert cpplink._cpplink.cli_version == cpplink.__version__


def test_help_prints_usage() -> None:
    result = cpplink.run(["--help"])
    assert result.code == 0
    assert result.stdout.startswith("usage: cpplink <command>")
    assert result.stderr == ""


def test_unknown_command_is_an_error_on_stderr() -> None:
    result = cpplink.run(["frobnicate"])
    assert result.code == 1
    assert "unknown command 'frobnicate'" in result.stderr
    assert result.stdout == ""


def test_missing_value_is_reported() -> None:
    result = cpplink.run(["estimate", "--schema"])
    assert result.code == 1
    assert "--schema needs a value" in result.stderr


def test_python_m_cpplink_forwards_argv() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cpplink", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == cpplink.__version__
    completed = subprocess.run(
        [sys.executable, "-m", "cpplink", "nope"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "unknown command" in completed.stderr
