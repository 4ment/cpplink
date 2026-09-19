# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""``python -m cpplink <command> [options]``: the command line, in process."""

from __future__ import annotations

import sys

from . import run


def main(argv: list[str] | None = None) -> int:
    result = run(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    return result.code


if __name__ == "__main__":
    sys.exit(main())
