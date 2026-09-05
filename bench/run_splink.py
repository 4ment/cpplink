# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run the splink deduplication pipeline on one prepared dataset.

The settings are compiled from the same registry cpplink's schema is compiled
from, so the comparison levels, the blocking and lambda are identical. What is
left to differ is how each tool estimates m and u, how it scores, and what it
costs.

Peak resident set is getrusage(RUSAGE_SELF): duckdb runs in this process, so
that is the whole pipeline's memory unless duckdb spills, which the report
notes by recording the temp directory's size.
"""

import argparse
import json
import os
import resource
import sys
import time
from contextlib import contextmanager

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS, LAMBDA_RECALL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RSS_SCALE = 1 if sys.platform == "darwin" else 1024


def build_comparison(comparison):
    """The same null -> exact -> fuzzy -> else ladder the cpplink schema gets."""
    levels = [
        cll.NullLevel(comparison.column),
        cll.ExactMatchLevel(
            comparison.column, term_frequency_adjustments=comparison.term_frequency
        ),
    ]
    for level in comparison.levels:
        if level.kind == "jw":
            levels.append(cll.JaroWinklerLevel(comparison.column, level.threshold))
        elif level.kind == "lev":
            levels.append(cll.LevenshteinLevel(comparison.column, int(level.threshold)))
        else:
            raise SystemExit(f"unknown level kind {level.kind!r}")
    levels.append(cll.ElseLevel())
    return cl.CustomComparison(
        output_column_name=comparison.column, comparison_levels=levels
    )


def build_settings(dataset, track):
    # Track "native" is cpplink's automatic blocking, which splink has no
    # counterpart for; splink keeps the declared rules in both tracks, which is
    # exactly the difference the native track is there to measure.
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[build_comparison(c) for c in dataset.comparisons],
        blocking_rules_to_generate_predictions=[
            block_on(c) for c in dataset.block_columns
        ],
        probability_two_random_records_match=dataset.lam,
        retain_matching_columns=False,
        retain_intermediate_calculation_columns=False,
    )


class Timer:
    def __init__(self):
        self.stages = {}

    @contextmanager
    def __call__(self, stage):
        start = time.perf_counter()
        yield
        self.stages[stage] = self.stages.get(stage, 0.0) + time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("--track", default="matched")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="0.1,0.5,0.9,0.99,0.999,0.9999,0.99999")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--estimate-lambda", action="store_true",
                        help="print lambda from deterministic rules and exit")
    parser.add_argument("--count-comparisons", action="store_true",
                        help="price the blocking rules and exit")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    parquet = os.path.join(DATA, dataset.name + ".parquet")
    os.makedirs(args.out, exist_ok=True)

    timer = Timer()
    with timer("load"):
        db_api = DuckDBAPI()
        connection = db_api._con
        if args.threads:
            connection.execute(f"PRAGMA threads={args.threads}")
        # A real table rather than a path: splink interpolates the input name
        # into SQL unquoted, and a path is not an identifier.
        connection.execute(
            "CREATE OR REPLACE TABLE bench_input AS "
            f"SELECT * FROM read_parquet('{parquet}')"
        )
        linker = Linker("bench_input", build_settings(dataset, args.track),
                        db_api=db_api)

    if args.count_comparisons:
        from splink.blocking_analysis import (
            count_comparisons_from_blocking_rule as count,
        )
        counts = {}
        for column in dataset.block_columns:
            result = count(table_or_tables="bench_input",
                           blocking_rule=block_on(column),
                           link_type="dedupe_only", db_api=db_api)
            counts[column] = {k: int(v) for k, v in result.items()
                              if isinstance(v, (int, float))}
        print(json.dumps(counts, indent=2))
        return

    if args.estimate_lambda:
        rules = [block_on(*columns) for columns in dataset.lambda_rules]
        linker.training.estimate_probability_two_random_records_match(
            rules, recall=LAMBDA_RECALL
        )
        print(json.dumps({
            "dataset": dataset.name,
            "lambda": linker._settings_obj._probability_two_random_records_match,
        }, indent=2))
        return

    with timer("estimate_u"):
        linker.training.estimate_u_using_random_sampling(
            max_pairs=args.u_sample, seed=args.seed
        )
    for column in dataset.block_columns:
        with timer("estimate_m"):
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on(column)
            )
    model = linker.misc.save_model_to_json()
    with open(os.path.join(args.out, "model.json"), "w") as handle:
        json.dump(model, handle, indent=2)

    with timer("predict"):
        predictions = linker.inference.predict(
            threshold_match_probability=min(thresholds)
        )
        edges = predictions.as_pandas_dataframe()
        edge_count = len(edges)

    clusters = {}
    for threshold in thresholds:
        with timer(f"cluster@{threshold}"):
            result = linker.clustering.cluster_pairwise_predictions_at_threshold(
                predictions, threshold_match_probability=threshold
            )
            frame = result.as_pandas_dataframe()[["unique_id", "cluster_id"]]
        path = os.path.join(args.out, f"clusters_p{threshold}.csv")
        frame.to_csv(path, index=False)
        clusters[str(threshold)] = path

    stages = timer.stages
    cluster_once = stages[f"cluster@{thresholds[0]}"]
    report = {
        "tool": "splink",
        "dataset": dataset.name,
        "track": args.track,
        "threads": args.threads,
        "stages": stages,
        "pipeline_seconds": (
            stages["load"] + stages["estimate_u"] + stages["estimate_m"]
            + stages["predict"] + cluster_once
        ),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_SCALE,
        "edges": edge_count,
        # Recorded rather than assumed: splink's EM reports a session-level
        # lambda that is biased upward by blocking, and it must not be the one
        # that ends up in the model if lambda is to stay held fixed.
        "lambda": model["probability_two_random_records_match"],
        "clusters": clusters,
    }
    with open(os.path.join(args.out, "timings.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
