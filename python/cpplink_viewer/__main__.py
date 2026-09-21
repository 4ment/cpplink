# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""``python -m cpplink_viewer [options] data.parquet``: the viewer, served."""

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
