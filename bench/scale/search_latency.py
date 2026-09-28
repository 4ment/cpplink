# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Per-query latency of `cpplink search`, swept over store size and thread count.

The question the sweep answers is which of the two phases dominates, because that
is what decides whether the next step is an inverted index or a q-gram index over
the dictionary. Phase one walks each queried column's dictionary once, so it grows
with the number of distinct values; phase two gathers a level per row, so it grows
with the rows. They are reported separately.

Three queries are run at every size, because a query's cost is a property of the
columns it names and not of the store:

    exact       a near-unique column with no fuzzy level reached (postcode, dob)
    fuzzy       one name column, which is a dictionary walk with a metric in it
    full        a whole record, which is every column the schema compares

Usage: search_latency.py <workdir> <rows> [<rows> ...]
It writes <workdir>/search_latency.json and prints the table. The workdir holds
s<rows>.parquet and s<rows>.model.json for every size; `gen-sample` writes the
first and `estimate` the second.

The recorded sweep is search_latency.json beside this file: 250k, 1M, 4M and 20M,
one thread and eight. The 20M rows there are single runs rather than the median of
five, because the store is 4.5 GB and reloading it per repeat is most of the wall
clock.
"""

import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPPLINK = ROOT / "build" / "cpplink"
SCHEMA = ROOT / "examples" / "sample_schema.json"

# The query record is row r0 of every `gen-sample` file, which is generated from
# the same seed at every size, so the three queries name the same person
# throughout and the only thing changing is how much data it is asked about.
QUERIES = {
    "exact": ["dob=1979-08-06", "postcode=4508"],
    "fuzzy": ["last_name=chirdloackki"],
    "full": [
        "first_name=zoudcut",
        "last_name=chirdloackki",
        "gender=F",
        "dob=1979-08-06",
        "email=zoudcut.chirdloackki3970@example.com",
        "phone=0404342668",
        "postcode=4508",
    ],
}
REPEATS = 5


def run(work, rows, name, fields, threads):
    path = work / f"s{rows}.parquet"
    model = work / f"s{rows}.model.json"
    command = [
        str(CPPLINK),
        "search",
        "--schema",
        str(SCHEMA),
        "--model",
        str(model),
        str(path),
        "--threads",
        str(threads),
        "-k",
        "10",
        "--json",
    ]
    for field in fields:
        command += ["--field", field]
    walks, gathers = [], []
    report = {}
    for _ in range(REPEATS):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        walks.append(report["walk_seconds"])
        gathers.append(report["gather_seconds"])
    return {
        "rows": rows,
        "query": name,
        "threads": threads,
        "walk_ms": 1000 * statistics.median(walks),
        "gather_ms": 1000 * statistics.median(gathers),
        "values_walked": report["values_walked"],
        "rescored": report["rescored"],
        "tabulated": report["tabulated"],
        "evaluated": report["evaluated"],
        "constant": report["constant"],
        "top": report["hits"][0]["id"] if report["hits"] else None,
        "top_weight": report["hits"][0]["match_weight"] if report["hits"] else None,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    work = pathlib.Path(sys.argv[1])
    sizes = [int(value) for value in sys.argv[2:]]
    rows = []
    for size in sizes:
        for name, fields in QUERIES.items():
            for threads in (1, 8):
                rows.append(run(work, size, name, fields, threads))
    (work / "search_latency.json").write_text(json.dumps(rows, indent=2))

    header = f"{'rows':>12}  {'query':<6} {'thr':>4} {'walk ms':>9} {'gather ms':>10}"
    header += f" {'total ms':>9} {'values':>12} {'rescored':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rows']:>12,}  {row['query']:<6} {row['threads']:>4}"
            f" {row['walk_ms']:>9.2f} {row['gather_ms']:>10.2f}"
            f" {row['walk_ms'] + row['gather_ms']:>9.2f}"
            f" {row['values_walked']:>12,} {row['rescored']:>10,}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
