# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Run the transactions linkage benchmark and report it.

For every track x tool, runs the pipeline `--repeat` times and keeps the median
wall clock and the largest peak resident set, then scores the predictions of
the first repeat at every threshold with the shared scorer.

    python bench/link/bench.py --verify           # matched candidate sets agree
    python bench/link/bench.py --repeat 3 --threads 1
    python bench/link/bench.py --tracks matched --tools cpplink --repeat 1
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score  # noqa: E402
from transactions import (  # noqa: E402
    CPPLINK_TRACKS,
    DESTINATION,
    MATCHED_KEYS,
    ORIGIN,
    RESULTS,
    ROWS,
    SPLINK_TRACKS,
    THRESHOLDS,
    TRACKS,
    schema_path,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.path.join(ROOT, "build", "cpplink")
PYTHON = sys.executable
RUNNERS = {
    "cpplink": os.path.join(HERE, "run_cpplink.py"),
    "splink": os.path.join(HERE, "run_splink.py"),
}


def human_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024


def verify():
    """The matched track's premise: both tools count the same pairs per key.

    cpplink's count comes from grouping the interned rows per input, splink's
    from a SQL join on the same column, so agreement is evidence that the two
    read the key the same way and not a tautology.
    """
    text = subprocess.run(
        [
            BINARY,
            "explain-blocking",
            "--schema",
            schema_path("matched"),
            "--count",
            ORIGIN,
            DESTINATION,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ours = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "exact_value":
            ours[parts[0]] = int(parts[-2].replace(",", ""))
    theirs = json.loads(
        subprocess.run(
            [
                PYTHON,
                RUNNERS["splink"],
                "--track",
                "matched",
                "--out",
                os.path.join(RESULTS, "verify"),
                "--count-comparisons",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    agree = True
    for key in MATCHED_KEYS:
        same = ours.get(key) == theirs.get(key)
        agree &= same
        print(
            f"  {key:<18} cpplink {ours.get(key):>12,}   splink {theirs.get(key):>12,}"
            f"   {'ok' if same else 'MISMATCH'}"
        )
    return agree


def run_once(tool, track, out, threads, repeat_index):
    command = [
        PYTHON,
        RUNNERS[tool],
        "--track",
        track,
        "--out",
        out,
        "--threads",
        str(threads),
        "--thresholds",
        THRESHOLDS,
    ]
    if tool == "cpplink" and repeat_index == 0:
        command.append("--analysis")
    subprocess.run(command, check=True, capture_output=True, text=True)
    with open(os.path.join(out, "timings.json")) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--tools", default="cpplink,splink")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    thresholds = [float(t) for t in THRESHOLDS.split(",")]

    if args.verify:
        ok = verify()
        print(
            "  matched blocking agrees"
            if ok
            else "  MISMATCH: the matched track is not matched"
        )
        return 0 if ok else 1

    summary = {"threads": args.threads, "repeat": args.repeat, "runs": []}
    for track in args.tracks.split(","):
        for tool in args.tools.split(","):
            if tool == "splink" and track not in SPLINK_TRACKS:
                continue
            if tool == "cpplink" and track not in CPPLINK_TRACKS:
                continue
            print(f"{tool} / {track}", flush=True)
            reports = []
            for repeat in range(args.repeat):
                out = os.path.join(RESULTS, f"{tool}.{track}.{repeat}")
                reports.append(run_once(tool, track, out, args.threads, repeat))
                print(
                    f"  repeat {repeat}: {reports[-1]['pipeline_seconds']:.2f} s, "
                    f"{human_bytes(reports[-1]['peak_rss_bytes'])}",
                    flush=True,
                )
            first = reports[0]
            run = {
                "tool": tool,
                "track": track,
                "pipeline_seconds": statistics.median(
                    r["pipeline_seconds"] for r in reports
                ),
                # The analysis stages run on the first repeat only.
                "stages": {
                    k: statistics.median(r["stages"][k] for r in reports)
                    for k in first["stages"]
                    if all(k in r["stages"] for r in reports)
                },
                "peak_rss_bytes": max(r["peak_rss_bytes"] for r in reports),
                "predictions": first["predictions"],
                "lambda": first["lambda"],
                "analysis": first.get("analysis", {}),
                "quality": score(first["predictions_path"], ROWS, thresholds),
            }
            run["best"] = max(run["quality"], key=lambda q: q["f1"])
            summary["runs"].append(run)

    with open(os.path.join(RESULTS, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n| track | tool | thr | predicted | precision | recall | F1 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for run in summary["runs"]:
        b = run["best"]
        print(
            f"| {run['track']} | {run['tool']} | {b['threshold']} | {b['predicted']:,} "
            f"| {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} |"
        )

    print("\n| track | tool | " + " | ".join(str(t) for t in thresholds) + " |")
    print("|---|---|" + "---:|" * len(thresholds))
    for run in summary["runs"]:
        print(
            f"| {run['track']} | {run['tool']} | "
            + " | ".join(f"{q['f1']:.4f}" for q in run["quality"])
            + " |"
        )

    print("\n| track | tool | candidates | blocking recall | pipeline s | peak RSS |")
    print("|---|---|---:|---:|---:|---:|")
    for run in summary["runs"]:
        a = run["analysis"]
        candidates = f"{a['candidate_pairs']:,}" if a else "-"
        recall = f"{a['blocking_recall']:.4f}" if a else "-"
        print(
            f"| {run['track']} | {run['tool']} | {candidates} | {recall} "
            f"| {run['pipeline_seconds']:.2f} | {human_bytes(run['peak_rss_bytes'])} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
