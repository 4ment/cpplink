# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run the benchmark and report it.

For every dataset x track x tool, runs the pipeline `--repeat` times and keeps
the median wall clock, then scores the cluster files from the first repeat with
the shared scorer. Quality is deterministic given the seed, so it is scored
once; only the cost is repeated.

    python bench.py --verify            # check the matched track really matches
    python bench.py                     # everything, 3 repeats
    python bench.py --datasets fake_1000 --tracks matched --repeat 1
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS, TRACKS  # noqa: E402
from score import score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PYTHON = sys.executable
RUNNERS = {
    "cpplink": os.path.join(HERE, "run_cpplink.py"),
    "splink": os.path.join(HERE, "run_splink.py"),
}


def human_bytes(count):
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024


def verify_blocking(dataset):
    """The matched track's premise: both tools consider the same candidates.

    cpplink's count is closed form from the term frequencies and splink's is a
    group-by, so agreement to the pair is meaningful rather than tautological.
    """
    from splink import DuckDBAPI, block_on
    from splink.blocking_analysis import count_comparisons_from_blocking_rule

    parquet = os.path.join(HERE, "data", dataset.name + ".parquet")
    schema = os.path.join(HERE, "schemas", f"{dataset.name}.matched.json")
    binary = os.path.join(os.path.dirname(HERE), "build", "cpplink")
    text = subprocess.run(
        [binary, "explain-blocking", "--schema", schema, "--count", parquet],
        capture_output=True, text=True, check=True,
    ).stdout

    ours = {}
    for line in text.splitlines():
        for column in dataset.block_columns:
            if line.startswith(column + " exact_value"):
                ours[column] = int(line.split()[-2].replace(",", ""))

    db_api = DuckDBAPI()
    db_api._con.execute(
        f"CREATE OR REPLACE TABLE bench_input AS SELECT * FROM read_parquet('{parquet}')"
    )
    agree = True
    for column in dataset.block_columns:
        theirs = count_comparisons_from_blocking_rule(
            table_or_tables="bench_input", blocking_rule=block_on(column),
            link_type="dedupe_only", db_api=db_api,
        )["number_of_comparisons_to_be_scored_post_filter_conditions"]
        same = ours.get(column) == theirs
        agree &= same
        print(f"  {column:<20} cpplink {ours.get(column):>12,}   "
              f"splink {theirs:>12,}   {'ok' if same else 'DIFFER'}")
    return agree


def run_one(tool, dataset, track, out, args):
    command = [PYTHON, RUNNERS[tool], dataset.name, "--track", track, "--out", out,
               "--thresholds", args.thresholds, "--threads", str(args.threads),
               "--u-sample", str(args.u_sample), "--seed", str(args.seed)]
    if tool == "cpplink":
        command.append("--analysis")
    os.makedirs(out, exist_ok=True)
    start = time.perf_counter()
    with open(os.path.join(out, "run.log"), "w") as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise SystemExit(f"{tool} {dataset.name}/{track} failed; see {out}/run.log")
    with open(os.path.join(out, "timings.json")) as handle:
        report = json.load(handle)
    report["wall_seconds"] = time.perf_counter() - start  # includes interpreter start
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--tracks", nargs="*", default=list(TRACKS))
    parser.add_argument("--tools", nargs="*", default=["cpplink", "splink"])
    parser.add_argument("--thresholds", default="0.1,0.5,0.9,0.99,0.999,0.9999,0.99999")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--u-sample", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--verify", action="store_true",
                        help="check candidate-set parity and exit")
    args = parser.parse_args()

    if args.verify:
        ok = True
        for name in args.datasets:
            print(f"{name}:")
            ok &= verify_blocking(DATASETS[name])
        print("\nmatched track is apples-to-apples" if ok
              else "\nCANDIDATE SETS DIFFER -- the matched track is not comparable")
        raise SystemExit(0 if ok else 1)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    rows = []
    for name in args.datasets:
        dataset = DATASETS[name]
        entities = os.path.join(HERE, "data", name + ".entities.csv")
        for track in args.tracks:
            for tool in args.tools:
                # splink has no automatic blocking, so its native-track
                # configuration is identical to its matched-track one. It is
                # run anyway: the two rows are then a repeatability check, and
                # each track's table is self-contained.
                reports = []
                for repeat in range(args.repeat):
                    out = os.path.join(RESULTS, name, track, tool, f"rep{repeat}")
                    print(f"[{name}/{track}] {tool} rep{repeat} ... ", end="", flush=True)
                    report = run_one(tool, dataset, track, out, args)
                    print(f"{report['pipeline_seconds']:.2f} s")
                    reports.append(report)
                first = reports[0]
                quality = {}
                for threshold in thresholds:
                    path = first["clusters"][str(threshold)]
                    quality[str(threshold)] = score(entities, path)
                best = max(quality.items(), key=lambda kv: kv[1]["f1"])
                if best[0] in (str(thresholds[0]), str(thresholds[-1])):
                    print(f"  ! best F1 is at the edge of the threshold grid "
                          f"({best[0]}); widen --thresholds")
                rows.append({
                    "dataset": name, "track": track, "tool": tool,
                    "pipeline_seconds": statistics.median(
                        r["pipeline_seconds"] for r in reports),
                    "wall_seconds": statistics.median(
                        r["wall_seconds"] for r in reports),
                    "peak_rss_bytes": max(r["peak_rss_bytes"] for r in reports),
                    "stages": first["stages"],
                    "quality": quality,
                    "best_threshold": best[0],
                    "best": best[1],
                })

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "summary.json"), "w") as handle:
        json.dump({"config": vars(args), "rows": rows}, handle, indent=2)

    print("\n## Quality at the best threshold\n")
    print("| dataset | track | tool | thr | precision | recall | F1 | clusters |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        b = row["best"]
        print(f"| {row['dataset']} | {row['track']} | {row['tool']} | "
              f"{row['best_threshold']} | {b['precision']:.4f} | {b['recall']:.4f} | "
              f"{b['f1']:.4f} | {b['clusters']:,} |")

    print("\n## Cost (median of "
          f"{args.repeat}, threads={args.threads or 'auto'})\n")
    print("| dataset | track | tool | pipeline s | peak RSS |")
    print("|---|---|---|---:|---:|")
    for row in rows:
        print(f"| {row['dataset']} | {row['track']} | {row['tool']} | "
              f"{row['pipeline_seconds']:.2f} | {human_bytes(row['peak_rss_bytes'])} |")

    print("\n## Threshold sweep (F1)\n")
    header = " | ".join(str(t) for t in thresholds)
    print(f"| dataset | track | tool | {header} |")
    print("|---|---|---|" + "---:|" * len(thresholds))
    for row in rows:
        cells = " | ".join(f"{row['quality'][str(t)]['f1']:.4f}" for t in thresholds)
        print(f"| {row['dataset']} | {row['track']} | {row['tool']} | {cells} |")
    print(f"\nFull results in {os.path.join(RESULTS, 'summary.json')}")


if __name__ == "__main__":
    main()
