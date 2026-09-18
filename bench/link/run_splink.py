# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run splink's transactions linkage on the prepared tables.

Track `demo` is splink's documentation example as written: the six SQL
blocking rules including `block_on("unique_id")`, the directed date levels,
u from a million random pairs, EM on memo then on amount, predictions at
0.001. Track `matched` swaps the rules for the precomputed keys and drops the
cheat, which is the configuration the candidate sets can be verified identical
to cpplink's under; the levels and the sessions are the demo's. Track
`symmetric` is `matched` with the date window either side.

Writes predictions.csv (origin_id, destination_id, match_probability) and
timings.json into --out. Peak resident set is getrusage(RUSAGE_SELF), since
duckdb runs in this process.
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
from transactions import (  # noqa: E402
    AMOUNT_PERCENTAGES, DATE_DAYS, DEMO_RULES_SQL, DESTINATION, LAMBDA, MATCHED_KEYS,
    MEMO_EDITS, ORIGIN, THRESHOLDS,
)

RSS_SCALE = 1 if sys.platform == "darwin" else 1024


def amount_comparison():
    return {
        "output_column_name": "amount",
        "comparison_levels": [
            cll.NullLevel("amount"),
            cll.ExactMatchLevel("amount"),
            *[cll.PercentageDifferenceLevel("amount", p) for p in AMOUNT_PERCENTAGES],
            cll.ElseLevel(),
        ],
        "comparison_description": "Amount percentage difference",
    }


def date_comparison(directed):
    # The demo's: destination on or after origin, within n days, which is
    # cpplink's `date_within` with `"direction": "forward"`. The symmetric
    # track's: within n days either way.
    if directed:
        template = ("transaction_date_r - transaction_date_l <= {n} "
                    "and transaction_date_r >= transaction_date_l")
    else:
        template = "abs(transaction_date_r - transaction_date_l) <= {n}"
    levels = [cll.NullLevel("transaction_date")]
    for n in DATE_DAYS:
        levels.append({"sql_condition": template.format(n=n),
                       "label_for_charts": f"<={n} days"})
    levels.append(cll.ElseLevel())
    return {
        "output_column_name": "transaction_date",
        "comparison_levels": levels,
        "comparison_description": "Transaction date days apart",
    }


def build_settings(track):
    if track == "demo":
        rules = list(DEMO_RULES_SQL)
    else:
        rules = [block_on(key) for key in MATCHED_KEYS]
    return SettingsCreator(
        link_type="link_only",
        probability_two_random_records_match=LAMBDA,
        blocking_rules_to_generate_predictions=rules,
        comparisons=[
            amount_comparison(),
            cl.LevenshteinAtThresholds("memo", MEMO_EDITS),
            date_comparison(directed=track != "symmetric"),
        ],
        # The demo sets this true, for its dashboards; it widens every row of
        # the prediction table and is not part of either model.
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
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
    parser.add_argument("--track", default="demo",
                        choices=("demo", "matched", "symmetric"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default=THRESHOLDS)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--count-comparisons", action="store_true",
                        help="price each blocking rule and exit")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out, exist_ok=True)

    timer = Timer()
    with timer("load"):
        db_api = DuckDBAPI()
        connection = db_api._con
        if args.threads:
            connection.execute(f"PRAGMA threads={args.threads}")
        for name, path in (("origin", ORIGIN), ("destination", DESTINATION)):
            connection.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{path}')"
            )
        # Aliases sorted so that origin is `l` and destination is `r`, as the
        # demo arranges with "__ori" and "_dest": the directed date levels and
        # the two asymmetric rules read the sides by name.
        linker = Linker(["origin", "destination"], build_settings(args.track),
                        input_table_aliases=["__ori", "_dest"], db_api=db_api)

    if args.count_comparisons:
        from splink.blocking_analysis import (
            count_comparisons_from_blocking_rule as count,
        )
        rules = (DEMO_RULES_SQL if args.track == "demo"
                 else [block_on(k) for k in MATCHED_KEYS])
        names = ([f"rule_{i}" for i in range(len(rules))] if args.track == "demo"
                 else MATCHED_KEYS)
        counts = {}
        for name, rule in zip(names, rules):
            result = count(table_or_tables=["origin", "destination"],
                           blocking_rule=rule, link_type="link_only", db_api=db_api)
            counts[name] = int(
                result["number_of_comparisons_to_be_scored_post_filter_conditions"])
        print(json.dumps(counts, indent=2))
        return

    with timer("estimate_u"):
        linker.training.estimate_u_using_random_sampling(
            max_pairs=args.u_sample, seed=args.seed)
    # The demo's two sessions, in its order, in both tracks: each holds out the
    # comparison it blocks on and fits the other two.
    for column in ("memo", "amount"):
        with timer("estimate_m"):
            linker.training.estimate_parameters_using_expectation_maximisation(
                block_on(column))
    model = linker.misc.save_model_to_json()
    with open(os.path.join(args.out, "model.json"), "w") as handle:
        json.dump(model, handle, indent=2)

    with timer("predict"):
        predictions = linker.inference.predict(
            threshold_match_probability=min(thresholds))
        frame = predictions.as_pandas_dataframe()[
            ["source_dataset_l", "unique_id_l", "source_dataset_r", "unique_id_r",
             "match_probability"]]
    # Origin is always l here, but the file says so rather than assuming it.
    swap = frame["source_dataset_l"] != "__ori"
    origin = frame["unique_id_l"].where(~swap, frame["unique_id_r"])
    destination = frame["unique_id_r"].where(~swap, frame["unique_id_l"])
    out = frame.assign(origin_id=origin.astype(str),
                       destination_id=destination.astype(str))[
        ["origin_id", "destination_id", "match_probability"]]
    predictions_path = os.path.join(args.out, "predictions.csv")
    out.to_csv(predictions_path, index=False)

    stages = timer.stages
    report = {
        "tool": "splink",
        "track": args.track,
        "threads": args.threads,
        "stages": stages,
        "pipeline_seconds": stages["load"] + stages["estimate_u"]
        + stages["estimate_m"] + stages["predict"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_SCALE,
        "predictions": len(out),
        "lambda": model["probability_two_random_records_match"],
        "predictions_path": predictions_path,
    }
    with open(os.path.join(args.out, "timings.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
