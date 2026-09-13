# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Find and run the cpplink binary, for the numbers only it can produce.

The studio never reimplements what the binary reports: the exact blocking
report, the union count, the profile ledger, recall against a truth file and
the level proposal all come from here as parsed JSON. The binary is found in
``$CPPLINK``, then ``build/cpplink`` under the repository, then on ``PATH``.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class BinaryError(RuntimeError):
    pass


def find_binary():
    env = os.environ.get("CPPLINK")
    if env and os.access(env, os.X_OK):
        return env
    local = os.path.join(ROOT, "build", "cpplink")
    if os.access(local, os.X_OK):
        return local
    return shutil.which("cpplink")


def run(args, timeout=None, binary=None):
    """Run one command; (stdout, stderr, seconds). Raises on a non-zero exit."""
    binary = binary or find_binary()
    if binary is None:
        raise BinaryError("cpplink binary not found: set $CPPLINK or build it")
    started = time.time()
    proc = subprocess.run([binary] + list(args), capture_output=True, text=True,
                          timeout=timeout)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise BinaryError(proc.stderr.strip() or proc.stdout.strip() or
                          f"cpplink exited {proc.returncode}")
    return proc.stdout, proc.stderr, elapsed


def run_json(args, timeout=None, binary=None):
    out, err, elapsed = run(args, timeout, binary)
    try:
        return json.loads(out), err, elapsed
    except json.JSONDecodeError as exc:
        raise BinaryError(f"cpplink did not return JSON: {exc}\n{out[:500]}")


def write_schema_temp(schema_json):
    fd, path = tempfile.mkstemp(prefix="cpplink-studio-", suffix=".json")
    with os.fdopen(fd, "w") as handle:
        handle.write(schema_json)
    return path


def explain_blocking(schema_json, data_paths, count=False, mode=None, timeout=None):
    path = write_schema_temp(schema_json)
    try:
        args = ["explain-blocking", "--schema", path, "--json"]
        if count:
            args.append("--count")
        if mode:
            args += ["--mode", mode]
        return run_json(args + list(data_paths), timeout)
    finally:
        os.unlink(path)


def inspect(schema_json, data_paths, timeout=None):
    path = write_schema_temp(schema_json)
    try:
        return run(["inspect", "--schema", path] + list(data_paths), timeout)
    finally:
        os.unlink(path)


def profile(schema_json, data_paths, extra=(), timeout=None):
    path = write_schema_temp(schema_json)
    try:
        args = ["profile", "--schema", path, "--json"] + list(extra) + list(data_paths)
        return run_json(args, timeout)
    finally:
        os.unlink(path)


def recall(schema_json, data_paths, truth_path, count=False, timeout=None):
    path = write_schema_temp(schema_json)
    try:
        args = ["recall", "--schema", path, "--truth", truth_path, "--json"]
        if count:
            args.append("--count")
        return run_json(args + list(data_paths), timeout)
    finally:
        os.unlink(path)


def levels(schema_json, data_paths, extra=(), timeout=None):
    path = write_schema_temp(schema_json)
    try:
        args = ["levels", "--schema", path, "--json"] + list(extra) + list(data_paths)
        return run_json(args, timeout)
    finally:
        os.unlink(path)
