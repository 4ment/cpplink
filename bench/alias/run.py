# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""What the name/alias comparison shape buys, measured on `generate.py`'s data.

Two tracks, each holding everything fixed but one thing.

The comparison track runs one blocking plan, on the main columns only, under four
schemas: every name compared as a scalar or beside its alias list, each with
term frequency off or on. The 2x2 separates what the alias levels are worth from
what term frequency on the shape's exact level is worth.

The blocking track keeps the best of those comparisons and adds a key per field
built from the main value and its aliases, so a record filed under its nickname
can still be blocked with one filed under the full name. Each key is the smallest
element of {main} union aliases, written into the file and declared with
`derived_from`, so estimation holds out the comparison the key was cut from.

Quality is the pairwise F1 of the clustering over its transitive closure, scored
by `bench/score.py` from the entities file, at every threshold in the sweep.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)
BINARY = os.path.join(ROOT, "build", "cpplink")
sys.path.insert(0, BENCH)
from score import score  # noqa: E402

THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
FIELDS = [
    # (comparison, main column, alias column, the fuzzy levels as (type, threshold))
    (
        "first_name",
        "first_name",
        "first_name_aliases",
        [("jaro_winkler", 0.92), ("jaro_winkler", 0.80)],
    ),
    (
        "surname",
        "surname",
        "surname_aliases",
        [("jaro_winkler", 0.92), ("jaro_winkler", 0.80)],
    ),
    ("dob", "dob", "dob_aliases", [("levenshtein", 1), ("levenshtein", 2)]),
]
CONTAINS = {
    "jaro_winkler": "contains_jaro_winkler",
    "levenshtein": "contains_levenshtein",
}


def levels_for(fuzzy, alias):
    """null, exact, [list_contains], fuzzy[0], [its contains twin], fuzzy[1:], else.

    Membership sits below an exact match and above a fuzzy one, and the fuzzy
    membership level below the scalar fuzzy level of the same threshold: a name
    one typo from the other's name is better evidence than a name one typo from
    one of its aliases.
    """
    levels = [{"type": "null"}, {"type": "exact"}]
    if alias:
        levels.append({"type": "list_contains"})
    for i, (kind, threshold) in enumerate(fuzzy):
        levels.append({"type": kind, "threshold": threshold})
        if alias and i == 0:
            levels.append({"type": CONTAINS[kind], "threshold": threshold})
    levels.append({"type": "else"})
    return levels


def schema(alias, tf, keys):
    columns = []
    comparisons = []
    for name, main, aliases, fuzzy in FIELDS:
        columns.append({"name": main, "type": "string"})
        if alias:
            columns.append({"name": aliases, "type": "string_list"})
        comparisons.append(
            {
                "name": name,
                "columns": [main, aliases] if alias else [main],
                "term_frequency": tf,
                "levels": levels_for(fuzzy, alias),
            }
        )
    columns.append({"name": "postcode", "type": "string"})
    comparisons.append(
        {
            "name": "postcode",
            "columns": ["postcode"],
            "term_frequency": tf,
            "levels": [
                {"type": "null"},
                {"type": "exact"},
                {"type": "levenshtein", "threshold": 1},
                {"type": "else"},
            ],
        }
    )
    blocking = [
        {"name": "dob", "type": "exact_value", "column": "dob"},
        {"name": "postcode", "type": "exact_value", "column": "postcode"},
        {
            "name": "surname",
            "type": "rare_value",
            "column": "surname",
            "max_frequency": 200,
        },
    ]
    if keys:
        for name, main, aliases, _ in FIELDS:
            key = f"{name}_key"
            columns.append(
                {"name": key, "type": "string", "derived_from": [main, aliases]}
            )
            if name == "dob":
                blocking.append({"name": key, "type": "exact_value", "column": key})
            else:
                blocking.append(
                    {
                        "name": key,
                        "type": "rare_value",
                        "column": key,
                        "max_frequency": 200,
                    }
                )
    return {
        "unique_id": "unique_id",
        "columns": columns,
        "comparisons": comparisons,
        "blocking": blocking,
    }


VARIANTS = [
    # name, alias levels, term frequency, alias keys in blocking
    ("scalar", False, False, False),
    ("scalar_tf", False, True, False),
    ("alias", True, False, False),
    ("alias_tf", True, True, False),
    ("alias_tf_keys", True, True, True),
]


def add_keys(source, target):
    """The smallest of each field's main value and its aliases, as a column."""
    table = pq.read_table(source)
    for name, main, aliases, _ in FIELDS:
        values = table.column(main).to_pylist()
        lists = table.column(aliases).to_pylist()
        keys = []
        for value, listed in zip(values, lists, strict=True):
            pool = [v for v in [value, *(listed or [])] if v]
            keys.append(min(pool) if pool else None)
        table = table.append_column(f"{name}_key", pa.array(keys, pa.string()))
    pq.write_table(table, target)
    return pc.count_distinct(table.column("first_name_key")).as_py()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="generate.py's output directory")
    parser.add_argument("--out", required=True, help="directory for this run")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--only", help="comma-separated variants to run")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = os.path.join(args.data, "alias.parquet")
    keyed = os.path.join(args.out, "alias.keyed.parquet")
    entities = os.path.join(args.data, "alias.entities.csv")
    truth = os.path.join(args.data, "alias.truth.csv")
    add_keys(data, keyed)

    def run(*command):
        started = time.perf_counter()
        done = subprocess.run(
            [BINARY, *map(str, command)], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise SystemExit(f"{command[0]} failed:\n{done.stdout}\n{done.stderr}")
        return done.stdout, time.perf_counter() - started

    wanted = set(args.only.split(",")) if args.only else None
    results = []
    for name, alias, tf, keys in VARIANTS:
        if wanted and name not in wanted:
            continue
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "schema.json")
        with open(path, "w") as handle:
            json.dump(schema(alias, tf, keys), handle, indent=2)
        parquet = keyed if keys else data
        model = os.path.join(out, "model.json")
        predictions = os.path.join(out, "predictions.parquet")

        reached, _ = run(
            "recall", "--schema", path, "--truth", truth, "--count", "--json", parquet
        )
        reached = json.loads(reached)
        _, estimate_s = run(
            "estimate",
            "--schema",
            path,
            "--out",
            model,
            "--threads",
            args.threads,
            "--seed",
            args.seed,
            parquet,
        )
        _, predict_s = run(
            "predict",
            "--schema",
            path,
            "--model",
            model,
            "--out",
            predictions,
            "--probability",
            min(THRESHOLDS),
            "--threads",
            args.threads,
            parquet,
        )
        sweep = []
        for threshold in THRESHOLDS:
            clusters = os.path.join(out, f"clusters_p{threshold}.csv")
            run(
                "cluster",
                "--schema",
                path,
                "--predictions",
                predictions,
                "--out",
                clusters,
                "--probability",
                threshold,
                "--min-size",
                2,
                parquet,
            )
            quality = score(entities, clusters)
            sweep.append(
                {
                    "probability": threshold,
                    **{key: quality[key] for key in ("precision", "recall", "f1")},
                }
            )
        best = max(sweep, key=lambda row: row["f1"])
        with open(model) as handle:
            learned = json.load(handle)
        result = {
            "variant": name,
            "alias_levels": alias,
            "term_frequency": tf,
            "alias_keys": keys,
            "candidates": reached["candidate_union"],
            "blocking_recall": reached["pair_completeness"],
            "lambda": learned["lambda"],
            "best": best,
            "sweep": sweep,
            "estimate_seconds": round(estimate_s, 2),
            "predict_seconds": round(predict_s, 2),
        }
        results.append(result)
        print(
            f"{name:<15} PC {result['blocking_recall']:.4f}  "
            f"candidates {result['candidates']:>10,}  best F1 {best['f1']:.4f} "
            f"(P {best['precision']:.4f}, R {best['recall']:.4f}, "
            f"p >= {best['probability']})",
            flush=True,
        )

    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
