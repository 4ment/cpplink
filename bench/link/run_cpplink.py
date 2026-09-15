# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run cpplink's transactions linkage on the prepared tables.

Two files, so every command is in link mode without being told. The model is
estimated from a Bernoulli sample of the whole cross product (`estimate
--all-pairs --session-pairs N`) rather than from blocked sessions, because a
session here would have fewer than the three free comparisons cpplink insists
on: the data has three comparisons and a blocked session holds one out. An
unblocked session holds nothing out, which is sound exactly when no two
comparison columns are tied, and amount, memo and date are not. The blocking
plan is then prediction's alone.

Writes predictions.csv (origin_id, destination_id, match_probability),
timings.json and the blocking analysis into --out. Peak resident set is
getrusage(RUSAGE_CHILDREN): cpplink is the only thing this process spawns.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transactions import (  # noqa: E402
    DESTINATION, LAMBDA, ORIGIN, THRESHOLDS, TRUTH, schema_path,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.path.join(ROOT, "build", "cpplink")
RSS_SCALE = 1 if sys.platform == "darwin" else 1024


class Runner:
    def __init__(self, log):
        self.log = log
        self.stages = {}

    def __call__(self, stage, *argv):
        command = [BINARY] + [str(a) for a in argv]
        self.log.write("$ " + " ".join(command) + "\n")
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        self.stages[stage] = self.stages.get(stage, 0.0) + time.perf_counter() - start
        self.log.write(result.stdout)
        self.log.write(result.stderr)
        self.log.flush()
        if result.returncode != 0:
            raise SystemExit(f"cpplink {stage} failed:\n{result.stderr}\n{result.stdout}")
        return result.stdout


def parse_blocking(text):
    """Per-source candidate counts and the deduplicated union, off the table."""
    sources = {}
    union = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] in ("exact_value", "sorted_neighbourhood",
                                            "rare_value"):
            sources[parts[0]] = int(parts[-2].replace(",", ""))
        if line.startswith("Union, deduplicated"):
            union = int(parts[-1].replace(",", ""))
    return sources, union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="matched", choices=("demo", "matched", "native"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default=THRESHOLDS)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--session-pairs", type=int, default=100000000,
                        help="pairs of the cross product compared for m")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--analysis", action="store_true",
                        help="also price the blocking and measure its recall")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    schema = schema_path(args.track)
    os.makedirs(args.out, exist_ok=True)
    model = os.path.join(args.out, "model.json")
    raw = os.path.join(args.out, "predictions.raw.csv")
    inputs = [ORIGIN, DESTINATION]

    with open(os.path.join(args.out, "cpplink.log"), "w") as log:
        run = Runner(log)
        analysis = {}
        if args.analysis:
            blocking = run("explain_blocking", "explain-blocking", "--schema", schema,
                           "--count", *inputs)
            sources, union = parse_blocking(blocking)
            reached = json.loads(run("recall", "recall", "--schema", schema, "--truth",
                                     TRUTH, "--count", "--json", *inputs))
            if union is not None and union != reached["candidate_union"]:
                raise SystemExit(f"blocking priced at {union:,} and enumerated at "
                                 f"{reached['candidate_union']:,}")
            analysis = {
                "sources": sources,
                "candidate_pairs": reached["candidate_union"],
                "truth_pairs": reached["truth_pairs"],
                "unresolved": reached["unresolved"],
                "truth_pairs_reached": reached["union_found"],
                "blocking_recall": reached["pair_completeness"],
                "pair_quality": reached["pair_quality"],
            }

        run("estimate", "estimate", "--schema", schema, "--out", model, "--all-pairs",
            "--session-pairs", args.session_pairs, "--u-sample", args.u_sample,
            "--lambda", repr(LAMBDA), "--threads", args.threads, "--seed", args.seed,
            *inputs)
        run("predict", "predict", "--schema", schema, "--model", model, "--out", raw,
            "--probability", repr(min(thresholds)), "--threads", args.threads, *inputs)

    frame = pd.read_csv(raw, dtype={"id_a": str, "id_b": str})
    # Link mode emits the earlier input first, and the file says which is which.
    swap = frame["dataset_a"] != "origin"
    origin = frame["id_a"].where(~swap, frame["id_b"])
    destination = frame["id_b"].where(~swap, frame["id_a"])
    out = frame.assign(origin_id=origin, destination_id=destination)[
        ["origin_id", "destination_id", "match_probability"]]
    predictions_path = os.path.join(args.out, "predictions.csv")
    out.to_csv(predictions_path, index=False)
    os.remove(raw)

    with open(model) as handle:
        learned = json.load(handle)
    stages = run.stages
    report = {
        "tool": "cpplink",
        "track": args.track,
        "threads": args.threads,
        "stages": stages,
        "pipeline_seconds": stages["estimate"] + stages["predict"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        * RSS_SCALE,
        "predictions": len(out),
        "lambda": learned["lambda"],
        "predictions_path": predictions_path,
        "analysis": analysis,
    }
    with open(os.path.join(args.out, "timings.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
