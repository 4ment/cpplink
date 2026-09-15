# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Materialise the two transactions tables in the form both tools read.

Writes into bench/data/transactions/:

  origin.parquet, destination.parquet
      unique_id as a string, memo as a string, transaction_date as a date,
      amount as a double, then one string column per blocking key computed by
      duckdb from the demo's SQL, and `amount_key`, the amount zero-padded so
      it sorts numerically. The `ground_truth` column is not written: neither
      tool may see the answer, and it equals unique_id anyway.
  truth.csv
      origin:<id>,destination:<id> for every row, qualified because the two
      files share every id.

and into bench/schemas/ one cpplink schema per track.
"""

import argparse
import csv
import json
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transactions import (  # noqa: E402
    CPPLINK_TRACKS, DATA, DESTINATION, KEYS, ORIGIN, ROWS, SCHEMAS, TRUTH,
    cpplink_schema, schema_path,
)


def write_side(connection, frame, path, side):
    """One table: the demo's columns typed, then the keys for this side."""
    connection.register("src", frame)
    keys = ",\n  ".join(
        f"({origin if side == 'origin' else destination}) AS {name}"
        for name, origin, destination in KEYS
    )
    connection.execute(f"""
        COPY (
          SELECT
            CAST(unique_id AS VARCHAR) AS unique_id,
            CAST(memo AS VARCHAR) AS memo,
            CAST(transaction_date AS DATE) AS transaction_date,
            CAST(amount AS DOUBLE) AS amount,
            {keys},
            printf('%016.2f', amount) AS amount_key
          FROM src
          ORDER BY CAST(unique_id AS INTEGER)
        ) TO '{path}' (FORMAT PARQUET)
    """)
    connection.unregister("src")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from splink.datasets import splink_datasets

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(SCHEMAS, exist_ok=True)
    origin = splink_datasets.transactions_origin
    destination = splink_datasets.transactions_destination
    for name, frame in (("origin", origin), ("destination", destination)):
        if len(frame) != ROWS:
            print(f"  ! {name}: expected {ROWS} rows, got {len(frame)}")
        if not frame["unique_id"].is_unique:
            raise SystemExit(f"{name}: unique_id is not unique")
        if not (frame["ground_truth"] == frame["unique_id"]).all():
            raise SystemExit(f"{name}: ground_truth is not unique_id; the truth "
                             "file below assumes it is")

    connection = duckdb.connect()
    write_side(connection, origin, ORIGIN, "origin")
    write_side(connection, destination, DESTINATION, "destination")

    with open(TRUTH, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_a", "id_b"])
        for uid in origin["unique_id"]:
            writer.writerow([f"origin:{uid}", f"destination:{uid}"])

    for track in CPPLINK_TRACKS:
        with open(schema_path(track), "w") as handle:
            json.dump(cpplink_schema(track), handle, indent=2)
            handle.write("\n")

    for name, path in (("origin", ORIGIN), ("destination", DESTINATION)):
        distinct = connection.execute(
            "SELECT " + ", ".join(f"count(DISTINCT {k[0]})" for k in KEYS)
            + f" FROM read_parquet('{path}')"
        ).fetchone()
        print(f"  {name}: {ROWS:,} rows; distinct keys "
              + ", ".join(f"{k[0]}={d:,}" for k, d in zip(KEYS, distinct)))
    print(f"  {ROWS:,} truth pairs, one per origin record")


if __name__ == "__main__":
    main()
