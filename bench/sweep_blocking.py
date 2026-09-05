# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Price every blocking method against every other, on one frontier.

`recall` answers "what does this plan reach"; this answers the question that
actually decides a default configuration: at a candidate budget, which method
reaches the most known pairs. Every method here can buy pair completeness with
candidates -- a sorted-neighbourhood window can be widened until it enumerates
the file -- so a single operating point per method compares nothing. Each
method is therefore swept over its own knob and reported as a curve.

Methods are run **in isolation**, one source per plan. A source measured inside
a plan is credited only with what earlier sources left it, which is the right
number for "should I add this to that plan" and the wrong one for "which method
is better". Both are wanted, so the sweep runs isolated curves first and then a
marginal pass that adds each automatic source to the declared plan.

The traditional baselines are the ones the record linkage literature has used
since Hernandez & Stolfo 1995: standard blocking (exact agreement on one
column, one run per column) and sorted neighbourhood. The automatic sources are
rare-value agreement and MinHash LSH. Nothing here is novel; the point of the
sweep is that the choice between them on this data is measured rather than
assumed.
"""

import argparse
import json
import os
import subprocess
import sys

from datasets import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
BINARY = os.path.join(ROOT, "build", "cpplink")
OUT = os.path.join(HERE, "results", "blocking")

# Knob grids. Each spans from "far too tight to be useful" to "far too loose to
# afford", because a frontier that does not bracket the knee cannot show one.
FREQUENCY_GRID = [2, 5, 10, 20, 50, 100, 200, 500]
WINDOW_GRID = [2, 4, 8, 16, 32, 64, 128]
BANDS_GRID = [2, 4, 8, 16, 32]
ROWS_PER_BAND = 4


def methods(dataset):
    """(family, label, knob, [blocking specs]) for every point on every curve.

    A method whose knob is a column rather than a number -- standard blocking --
    contributes one point per column, which is exactly how it is tuned in
    practice: you pick the column, not a threshold.
    """
    points = []
    for column in dataset.columns:
        points.append(("exact_value", column, None,
                       [{"type": "exact_value", "column": column}]))
    for column in dataset.columns:
        for frequency in FREQUENCY_GRID:
            points.append(("rare_value", column, frequency,
                           [{"type": "rare_value", "column": column,
                             "max_frequency": frequency}]))
        for window in WINDOW_GRID:
            points.append(("sorted_neighbourhood", column, window,
                           [{"type": "sorted_neighbourhood", "column": column,
                             "window": window}]))
        for bands in BANDS_GRID:
            points.append(("minhash", column, bands,
                           [{"type": "minhash", "column": column, "bands": bands,
                             "rows_per_band": ROWS_PER_BAND}]))
    return points


def run_recall(dataset, blocking, parquet, truth, scratch, count):
    """One `recall --json`, on a schema holding exactly the given sources."""
    schema = dataset.cpplink_schema("matched")
    schema["blocking"] = blocking
    path = os.path.join(scratch, "sweep.json")
    with open(path, "w") as handle:
        json.dump(schema, handle)
    command = [BINARY, "recall", "--schema", path, "--truth", truth, "--json"]
    if count:
        command.append("--count")
    command.append(parquet)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        # A method can be inapplicable to a column -- MinHash needs a string
        # column, and a plan can price a source at zero pairs. That is a result,
        # not a crash, so it is recorded rather than aborting the sweep.
        return {"error": done.stderr.strip()}
    return json.loads(done.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--count", action="store_true",
                        help="enumerate the deduplicated union instead of "
                             "bounding it by the sum over sources")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets.split(","):
        dataset = DATASETS[name]
        parquet = os.path.join(DATA, name + ".parquet")
        truth = os.path.join(DATA, name + ".truth.csv")
        if not os.path.exists(parquet):
            sys.exit(f"{parquet} is missing; run bench/prepare.py first")

        scratch = os.path.join(args.out, name)
        os.makedirs(scratch, exist_ok=True)
        rows = []

        # Isolated curves: what each method reaches on its own.
        for family, label, knob, blocking in methods(dataset):
            result = run_recall(dataset, blocking, parquet, truth, scratch,
                                args.count)
            rows.append({"pass": "isolated", "family": family, "column": label,
                         "knob": knob, **summarise(result)})
            print(f"{name} {family:22} {label:20} {str(knob):>5} "
                  f"{format_row(rows[-1])}", flush=True)

        # Marginal pass: what each automatic source adds to the declared plan.
        declared = dataset.cpplink_schema("matched")["blocking"]
        base = run_recall(dataset, declared, parquet, truth, scratch, args.count)
        rows.append({"pass": "plan", "family": "declared", "column": "-",
                     "knob": None, **summarise(base)})
        # Only the columns the plan already blocks on are skipped. Excluding
        # standard blocking wholesale would hide the cheapest fix there is: a
        # column that agrees *exactly* on the missed pairs wants an exact-value
        # source, not a similarity-based one, and on historical_50k that column
        # is the one the declared plan leaves out.
        blocked = {spec["column"] for spec in declared}
        for family, label, knob, blocking in methods(dataset):
            if label in blocked and family == "exact_value":
                continue
            result = run_recall(dataset, declared + blocking, parquet, truth,
                                scratch, args.count)
            rows.append({"pass": "marginal", "family": family, "column": label,
                         "knob": knob, **summarise(result)})

        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w") as handle:
            json.dump({"dataset": name, "rows": rows}, handle, indent=2)
        print(f"\nwrote {path}\n")
        report(name, rows)


def summarise(result):
    if "error" in result:
        return {"error": result["error"]}
    return {
        "pair_completeness": result["pair_completeness"],
        "pair_quality": result["pair_quality"],
        "reduction_ratio": result["reduction_ratio"],
        "candidates": result["candidates"],
        "truth_pairs": result["truth_pairs"],
    }


def format_row(row):
    if "error" in row:
        return "-- " + row["error"].splitlines()[0][:60]
    return (f"PC {row['pair_completeness']:.4f}  "
            f"PQ {row['pair_quality']:.6f}  "
            f"cand {row['candidates']:,}")


def report(name, rows):
    """The frontier: at each candidate budget, the method that reaches most."""
    usable = [r for r in rows if r["pass"] == "isolated" and "error" not in r]
    usable.sort(key=lambda r: r["candidates"])
    print(f"### {name}: isolated frontier\n")
    print("| candidates | PC | PQ | method | column | knob |")
    print("|---:|---:|---:|---|---|---:|")
    best = -1.0
    for row in usable:
        # Only the points that are not dominated: a method costing more than an
        # earlier one and reaching no more is off the frontier and is noise in a
        # table meant to support a choice.
        if row["pair_completeness"] <= best:
            continue
        best = row["pair_completeness"]
        print(f"| {row['candidates']:,} | {row['pair_completeness']:.4f} | "
              f"{row['pair_quality']:.6f} | {row['family']} | {row['column']} | "
              f"{'' if row['knob'] is None else row['knob']} |")
    print()


if __name__ == "__main__":
    main()
