# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run the cpplink deduplication pipeline on one prepared dataset.

Stages are timed separately and each writes its report to the run log. Peak
resident set is taken from getrusage(RUSAGE_CHILDREN), which for this process
means the largest cpplink stage, since cpplink is the only thing it spawns.

The threshold sweep costs one `predict` and one `cluster` per threshold: the
edge shards carry their weight, so re-clustering higher never re-scores.
"""

import argparse
import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
BINARY = os.path.join(ROOT, "build", "cpplink")
RSS_SCALE = 1 if sys.platform == "darwin" else 1024  # ru_maxrss unit


def peak_child_bytes():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * RSS_SCALE


def parse_analysis(blocking, reached):
    """Pull the two numbers that explain a recall difference out of the reports.

    Blocking recall is the ceiling on the pipeline's recall: a truth pair no
    source produces is never scored, so no threshold recovers it.
    """
    analysis = {}
    for line in blocking.splitlines():
        if line.startswith("Union, deduplicated"):
            analysis["candidate_pairs"] = int(line.split()[-1].replace(",", ""))
    for line in reached.splitlines():
        if line.startswith("Union"):
            parts = line.split()
            analysis["truth_pairs_reached"] = int(parts[-2].replace(",", ""))
            analysis["blocking_recall"] = float(parts[-1].rstrip("%")) / 100.0
    return analysis


class Runner:
    def __init__(self, log):
        self.log = log
        self.stages = {}

    def __call__(self, stage, *args):
        command = [BINARY, *[str(a) for a in args]]
        self.log.write(f"\n$ {' '.join(command)}\n")
        self.log.flush()
        start = time.perf_counter()
        done = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        self.log.write(done.stdout)
        if done.stderr:
            self.log.write("[stderr] " + done.stderr)
        self.log.flush()
        if done.returncode != 0:
            raise SystemExit(f"cpplink {stage} failed:\n{done.stderr}")
        self.stages[stage] = self.stages.get(stage, 0.0) + elapsed
        return done.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("--track", default="matched")
    parser.add_argument("--out", required=True, help="directory for this run")
    parser.add_argument("--thresholds", default="0.1,0.5,0.9,0.99,0.999,0.9999,0.99999")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--analysis", action="store_true",
                        help="also price blocking and measure its recall")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    parquet = os.path.join(DATA, dataset.name + ".parquet")
    schema = os.path.join(HERE, "schemas", f"{dataset.name}.{args.track}.json")
    truth = os.path.join(DATA, dataset.name + ".truth.csv")

    out = args.out
    edges = os.path.join(out, "edges")
    shutil.rmtree(edges, ignore_errors=True)
    os.makedirs(edges, exist_ok=True)
    model = os.path.join(out, "model.json")

    with open(os.path.join(out, "cpplink.log"), "w") as log:
        run = Runner(log)
        analysis = {}
        if args.analysis:
            blocking = run("explain_blocking", "explain-blocking", "--schema", schema,
                           "--count", parquet)
            reached = run("recall", "recall", "--schema", schema, "--truth", truth,
                          parquet)
            analysis = parse_analysis(blocking, reached)

        run("estimate", "estimate", "--schema", schema, "--out", model,
            "--threads", args.threads, "--u-sample", args.u_sample,
            "--lambda", repr(dataset.lam), "--seed", args.seed, parquet)

        run("predict", "predict", "--schema", schema, "--model", model,
            "--out", edges, "--probability", repr(min(thresholds)),
            "--threads", args.threads, parquet)

        clusters = {}
        for threshold in thresholds:
            path = os.path.join(out, f"clusters_p{threshold}.csv")
            run(f"cluster@{threshold}", "cluster", "--schema", schema,
                "--edges", edges, "--out", path, "--probability", repr(threshold),
                "--min-size", 2, parquet)
            clusters[str(threshold)] = path

    with open(model) as handle:
        learned_lambda = json.load(handle)["lambda"]

    stages = run.stages
    cluster_once = stages[f"cluster@{thresholds[0]}"]
    edge_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(edges, "*")))
    report = {
        "tool": "cpplink",
        "dataset": dataset.name,
        "track": args.track,
        "threads": args.threads,
        "stages": stages,
        "pipeline_seconds": stages["estimate"] + stages["predict"] + cluster_once,
        "peak_rss_bytes": peak_child_bytes(),
        "edge_bytes": edge_bytes,
        "lambda": learned_lambda,
        "clusters": clusters,
        "analysis": analysis,
    }
    with open(os.path.join(out, "timings.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
