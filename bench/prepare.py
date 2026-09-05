# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Materialise each benchmark dataset once, in the form both tools read.

Writes, per dataset, into bench/data/:

  <name>.parquet      unique_id plus the comparison columns, every column a
                      string, nothing else -- in particular no label column,
                      so neither tool can see the answer
  <name>.entities.csv unique_id,entity -- the ground truth, for the scorer
  <name>.truth.csv    id_a,id_b -- the same truth as pairs, for cpplink's
                      `recall` command

and, into bench/schemas/, one cpplink schema per dataset per blocking track.
"""

import argparse
import csv
import itertools
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS, TRACKS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCHEMAS = os.path.join(HERE, "schemas")


def as_string_column(series):
    """Everything is a string; empty and NaN become a real null."""
    out = series.astype(object).where(series.notna(), None)
    out = out.map(lambda v: None if v is None else str(v).strip())
    return out.map(lambda v: None if v in (None, "", "nan", "None") else v)


def prepare(dataset, force=False):
    from splink.datasets import splink_datasets

    parquet = os.path.join(DATA, dataset.name + ".parquet")
    frame = getattr(splink_datasets, dataset.source).copy()
    frame.columns = [c.strip() for c in frame.columns]

    if len(frame) != dataset.rows:
        print(f"  ! expected {dataset.rows} rows, got {len(frame)}")

    entity = dataset.entity(frame).astype(str)
    ids = as_string_column(frame[dataset.id_column])
    if ids.duplicated().any():
        raise SystemExit(f"{dataset.name}: {dataset.id_column} is not unique")

    columns = {"unique_id": ids}
    for name in dataset.columns:
        if name not in frame.columns:
            raise SystemExit(f"{dataset.name}: no column {name!r} in the source")
        columns[name] = as_string_column(frame[name])
    table = pa.Table.from_pandas(
        pd.DataFrame(columns), schema=pa.schema([(c, pa.string()) for c in columns])
    )
    pq.write_table(table, parquet)

    entities = os.path.join(DATA, dataset.name + ".entities.csv")
    pd.DataFrame({"unique_id": ids, "entity": entity}).to_csv(entities, index=False)

    # Truth as pairs: every within-entity pair. Quadratic in the largest
    # entity, which is fine at these sizes and is what cpplink's recall
    # harness wants.
    truth = os.path.join(DATA, dataset.name + ".truth.csv")
    groups = pd.DataFrame({"id": ids, "entity": entity}).groupby("entity")["id"]
    pairs = 0
    with open(truth, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_a", "id_b"])
        for _, members in groups:
            for a, b in itertools.combinations(sorted(members), 2):
                writer.writerow([a, b])
                pairs += 1

    for track in TRACKS:
        path = os.path.join(SCHEMAS, f"{dataset.name}.{track}.json")
        with open(path, "w") as handle:
            json.dump(dataset.cpplink_schema(track), handle, indent=2)
            handle.write("\n")

    sizes = pd.Series(entity).value_counts()
    print(
        f"  {len(frame):>7,} rows  {sizes.size:>6,} entities  "
        f"{pairs:>9,} truth pairs  largest {sizes.max()}"
    )
    return {"rows": len(frame), "entities": int(sizes.size), "truth_pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(SCHEMAS, exist_ok=True)
    summary = {}
    for name in args.datasets or list(DATASETS):
        print(f"{name}:")
        summary[name] = prepare(DATASETS[name])
    with open(os.path.join(DATA, "datasets.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
