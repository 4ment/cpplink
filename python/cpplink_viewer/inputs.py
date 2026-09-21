# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The run's files as the viewer reads them: schema, shards, truth, model, waterfalls.

Nothing here imports the compiled module. The viewer reads what a run wrote and
so runs against a checkout with no build, or against a wheel with no Arrow.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# The binary shard `predict` writes: `kEdgeMagic` and one `kEdgeBytes` record per
# prediction, (row a, row b, packed gamma, match weight). A test reads a shard the
# core wrote through this, so the two cannot drift silently.
EDGE_MAGIC = b"CPPLNKE1"
EDGE_DTYPE = np.dtype([("a", "<u4"), ("b", "<u4"), ("g", "<u4"), ("w", "<f8")])

# What every waterfall file carries before the per-comparison columns.
WATERFALL_FIXED = [
    "id_a",
    "id_b",
    "gamma",
    "prior",
    "match_weight",
    "match_probability",
    "bracket_low",
    "bracket_high",
    "threshold",
    "zone",
    "emitted",
]


def schema_columns(path):
    """The id column and the columns actually read from the file."""
    with open(path) as handle:
        schema = json.load(handle)
    id_column = schema.get("unique_id", "unique_id")
    columns = [c["name"] for c in schema.get("columns", []) if "derive" not in c]
    return id_column, columns


def cell(value):
    """One value as the page shows it: a string, a joined list, or nothing."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(cell(v) or "" for v in value) or None
    return str(value)


def read_truth(path, wanted_ids):
    """Group number per wanted id, from the known pairs.

    Being a duplicate is transitive, so the whole file has to be walked: two
    clustered records can be the same entity through a third the page never
    shows. It is walked as integers, though: both columns are dictionary encoded
    into one dictionary and the union-find runs over that, so a truth file of
    any size costs one entry per distinct id rather than one string.
    """
    with open(path) as handle:
        first = handle.readline().rstrip("\n").split(",")
    header = first[:1] in (["id_a"], ["unique_id"], ["a"])
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(
            column_names=["a", "b"], skip_rows=1 if header else 0
        ),
        convert_options=pacsv.ConvertOptions(
            column_types={"a": pa.string(), "b": pa.string()}
        ),
    )
    left = table.column("a").combine_chunks()
    right = table.column("b").combine_chunks()
    pairs = len(left)
    codes = pa.concat_arrays([left, right]).dictionary_encode()
    index = codes.indices.to_numpy(zero_copy_only=False)

    parent = list(range(len(codes.dictionary)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(index[:pairs].tolist(), index[pairs:].tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    want = pa.array(sorted(wanted_ids), type=pa.string())
    at = pc.index_in(want, value_set=codes.dictionary).to_pylist()
    return {
        uid: find(i) for uid, i in zip(want.to_pylist(), at, strict=True) if i is not None
    }


def read_model_levels(path):
    """What a ledger needs from the model file: labels and rates per level.

    A waterfall row names the level each comparison landed on; its label, m and
    u are the model's rather than the pair's, so the file does not repeat them
    and they are read from here once.
    """
    with open(path) as handle:
        model = json.load(handle)
    comparisons = [
        {
            "name": c["name"],
            "levels": [
                {"label": lvl.get("label", str(i)), "m": lvl.get("m"), "u": lvl.get("u")}
                for i, lvl in enumerate(c["levels"])
            ],
        }
        for c in model["comparisons"]
    ]
    interactions = [f'{t["left"]} x {t["right"]}' for t in model.get("interactions", [])]
    return {
        "records": model.get("records"),
        "comparisons": comparisons,
        "interactions": interactions,
    }


def ledger(row, levels):
    """A wide waterfall row expanded to the ledger `cpplink explain --json` writes.

    The prior, one step per comparison with a running total, the two-way
    corrections, then the totals and the pattern's bracket and zone. Nothing is
    recomputed: every number is the scorer's, read back.
    """
    running = row["prior"]
    steps = []
    for c in levels["comparisons"]:
        name = c["name"]
        level = row[f"{name}_level"]
        lvl = c["levels"][level] if level < len(c["levels"]) else {"label": str(level)}
        bits, tf = row[f"{name}_bits"], row[f"{name}_tf"]
        running += bits + tf
        steps.append(
            {
                "name": name,
                "level": level,
                "label": lvl["label"],
                "m": lvl.get("m"),
                "u": lvl.get("u"),
                "bits": bits,
                "tf": tf,
                "frequency": row[f"{name}_frequency"],
                "running": running,
            }
        )
    interactions = []
    for name in levels["interactions"]:
        bits = row[name.replace(" x ", "_x_") + "_bits"]
        running += bits
        interactions.append({"name": name, "bits": bits, "running": running})
    return {
        "prior": row["prior"],
        "records": levels["records"],
        "threshold": row["threshold"],
        "steps": steps,
        "interactions": interactions,
        "weight": row["match_weight"],
        "probability": row["match_probability"],
        "bracket": [row["bracket_low"], row["bracket_high"]],
        "zone": row["zone"],
        "emitted": bool(row["emitted"]),
    }
