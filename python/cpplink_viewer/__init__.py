# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""``cpplink_viewer``: the cluster viewer, served over a DuckDB cache of a run.

It reads what a run wrote and never the compiled module, so it runs from a
checkout with no build (``PYTHONPATH=python python -m cpplink_viewer``) as well
as from the wheel (``cpplink-viewer``). DuckDB is the ``viewer`` extra.
"""

from .cache import Options, open_cache
from .server import Viewer, main, make_server

__all__ = ["Options", "Viewer", "main", "make_server", "open_cache"]
