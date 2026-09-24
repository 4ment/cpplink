# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""`cpplink-finder`: a page that asks who in a file a person is.

One store, one model, one query at a time, over `cpplink.Linker.search`. The
form is the schema's own: a box per column the model compares, typed by what
that column holds.
"""

from __future__ import annotations

from .form import BLANK, Box, FormError, FormSpec, build_form, to_query
from .rows import RowFetcher
from .session import Finder, FinderError, Hit, Outcome
from .settings import default_threads, settings_path

__all__ = [
    "BLANK",
    "Box",
    "Finder",
    "FinderError",
    "FormError",
    "FormSpec",
    "Hit",
    "Outcome",
    "RowFetcher",
    "build_form",
    "default_threads",
    "settings_path",
    "to_query",
]
