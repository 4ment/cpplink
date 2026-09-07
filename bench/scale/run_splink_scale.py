# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run splink over the same parquet and the same parity schema cpplink used.

Comparisons, blocking and lambda are compiled from parity_schema.json and from
cpplink's learned model, so the two tools are given the same model and what is
left to differ is the execution. Peak duckdb spill is sampled from the temp
directory rather than assumed, because scratch disk is the resource the design
claim is actually about.
"""
import argparse
import json
import os
import resource
import threading
import time

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on


def build_comparison(spec):
    """The same null -> exact -> fuzzy -> else ladder the cpplink schema gets."""
    column = spec["name"]
    levels = [cll.NullLevel(column)]
    for level in spec["levels"]:
        kind = level["type"]
        if kind in ("null", "else"):
            continue
        if kind == "exact":
            levels.append(
                cll.ExactMatchLevel(
                    column,
                    term_frequency_adjustments=spec.get("term_frequency", False),
                )
            )
        elif kind == "jaro_winkler":
            levels.append(cll.JaroWinklerLevel(column, level["threshold"]))
        elif kind == "levenshtein":
            levels.append(cll.LevenshteinLevel(column, int(level["threshold"])))
        else:
            raise SystemExit(f"level {kind!r} has no bit-identical counterpart")
    levels.append(cll.ElseLevel())
    return cl.CustomComparison(output_column_name=column, comparison_levels=levels)


class SpillSampler(threading.Thread):
    """Poll the duckdb temp directory so peak spill is measured, not guessed."""

    def __init__(self, path, interval=0.5):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self.peak, self.stopped = 0, False

    def run(self):
        while not self.stopped:
            total = 0
            for root, _, files in os.walk(self.path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            self.peak = max(self.peak, total)
            time.sleep(self.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temp", required=True)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--max-temp", default="16GB")
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--threads", type=int, default=0)
    # Splink 4 can fold the comparison vectors into agreement-pattern counts once
    # and iterate EM over the counts, which is the same sufficient statistic
    # cpplink's histogram is. It is off by default in splink, so it is measured
    # as its own variant rather than assumed.
    parser.add_argument("--pattern-counts", action="store_true")
    args = parser.parse_args()

    schema = json.load(open(args.schema))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.temp, exist_ok=True)
    stages = {}

    def timed(name, thunk):
        start = time.perf_counter()
        value = thunk()
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - start
        print(f"  {name}: {stages[name]:.2f}s", flush=True)
        return value

    spill = SpillSampler(args.temp)
    spill.start()

    def load():
        db_api = DuckDBAPI()
        con = db_api._con
        con.execute(f"SET temp_directory='{args.temp}'")
        con.execute(f"SET max_temp_directory_size='{args.max_temp}'")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        if args.threads:
            con.execute(f"PRAGMA threads={args.threads}")
        columns = ["id"] + [c["name"] for c in schema["columns"]]
        con.execute(
            "CREATE OR REPLACE TABLE bench_input AS "
            f"SELECT {', '.join(columns)} FROM read_parquet('{args.parquet}')"
        )
        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=[build_comparison(c) for c in schema["comparisons"]],
            blocking_rules_to_generate_predictions=[
                block_on(b["column"]) for b in schema["blocking"]
            ],
            probability_two_random_records_match=args.lam,
            unique_id_column_name="id",
            retain_matching_columns=False,
            retain_intermediate_calculation_columns=False,
        )
        return Linker("bench_input", settings, db_api=db_api), db_api

    linker, db_api = timed("load", load)
    timed(
        "estimate_u",
        lambda: linker.training.estimate_u_using_random_sampling(
            max_pairs=args.u_sample, seed=7
        ),
    )
    for rule in schema["blocking"]:
        timed(
            "estimate_m",
            lambda rule=rule: (
                linker.training.estimate_parameters_using_expectation_maximisation(
                    block_on(rule["column"]),
                    estimate_without_term_frequencies=args.pattern_counts,
                )
            ),
        )

    predictions = timed(
        "predict",
        lambda: linker.inference.predict(threshold_match_probability=args.threshold),
    )
    edges = db_api._con.execute(
        f"select count(*) from {predictions.physical_name}"
    ).fetchone()[0]
    print(f"  edges: {edges:,}", flush=True)

    clusters = timed(
        "cluster",
        lambda: linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, threshold_match_probability=args.threshold
        ),
    )
    db_api._con.execute(
        f"copy (select id, cluster_id from {clusters.physical_name}) "
        f"to '{args.out}/clusters.csv' (header)"
    )

    spill.stopped = True
    spill.join(timeout=3)
    report = {
        "tool": "splink",
        "parquet": os.path.basename(args.parquet),
        "stages": stages,
        "pipeline_seconds": sum(stages.values()),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_spill_bytes": spill.peak,
        "edges": edges,
        "threshold": args.threshold,
        "pattern_counts": args.pattern_counts,
    }
    with open(os.path.join(args.out, "timings.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
