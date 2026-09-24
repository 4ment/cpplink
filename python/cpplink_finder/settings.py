# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""What the schema cannot say: preferences, kept beside it.

Everything about the *form* comes from the schema, so what is left is what a
schema has no opinion about -- how many hits to return, how many threads to
split the pass over, what a score has to clear, and the prior a search asks
under -- plus the two cosmetic things a column name does not settle: what a box
is called and whether to draw it at all.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping


def default_threads() -> int:
    """Every core, because the row pass is the floor on a query's latency.

    At twenty million records even a one-field query spends most of a second
    walking the rows, and threading is the only lever on it: the walk splits
    four to five ways over eight threads.
    """
    return max(1, os.cpu_count() or 1)


def defaults() -> dict:
    return {
        "k": 10,
        "threads": default_threads(),
        "min_probability": 0.0,
        # The prior stated as the number of records here expected to be the
        # person asked about, which is the question a search asks. A model's own
        # prior is the match rate over the pair space, which is a different one.
        "expected_matches": 1.0,
        "labels": {},
        "hidden": [],
    }


def settings_path(schema_path: str | os.PathLike[str]) -> pathlib.Path:
    """Beside the schema, because the settings are that dataset's.

    Not *in* the schema: the schema is the model's, and a model estimated under
    one is not invalidated by someone asking for more hits.
    """
    path = pathlib.Path(schema_path)
    return path.with_suffix(path.suffix + ".finder.json")


def load(schema_path: str | os.PathLike[str]) -> dict:
    """The saved settings over the defaults, or the defaults alone."""
    settings = defaults()
    path = settings_path(schema_path)
    if not path.exists():
        return settings
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return settings
    if isinstance(saved, Mapping):
        settings.update({k: v for k, v in saved.items() if k in settings})
    return settings


def save(schema_path: str | os.PathLike[str], settings: Mapping) -> pathlib.Path:
    """Keep them, so they are the next session's starting point."""
    kept = defaults()
    kept.update({k: v for k, v in settings.items() if k in kept})
    path = settings_path(schema_path)
    path.write_text(json.dumps(kept, indent=2) + "\n")
    return path
