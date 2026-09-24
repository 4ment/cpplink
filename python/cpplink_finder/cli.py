# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Entry point: open the page on a schema, a model and a parquet.

    cpplink-finder --schema schema.json --model model.json people.parquet

The three files are what a search needs and none of them is made here: the
schema says what the columns are, `estimate` wrote the model, and the parquet is
the file being searched. Given no arguments the page asks for them itself.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", nargs="?", help="the parquet file to search")
    ap.add_argument("--schema", help="the schema the model was estimated under")
    ap.add_argument("--model", help="model.json, as `estimate` wrote it")
    ap.add_argument("--port", type=int, default=8501)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        page,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
        "--",
    ]
    if args.data:
        command.append(args.data)
    if args.schema:
        command += ["--schema", args.schema]
    if args.model:
        command += ["--model", args.model]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
