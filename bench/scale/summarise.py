# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Collect one scale run into a single JSON and print the report's tables.

Timings are the minimum over repeats, which is the usual choice for a wall-clock
benchmark on a shared machine: the fastest run is the one least contaminated by
whatever else the operating system was doing.
"""
import argparse
import glob
import json
import os
import re

SIZES = [("s1m", 1_000_000), ("s2m", 2_000_000), ("s4m", 4_000_000)]
STAGES = ("estimate", "predict", "cluster")


def parse_time(path):
    text = open(path).read()
    wall = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    if not wall or not rss:
        return None
    return {
        "wall": float(wall.group(1)),
        "cpu": float(wall.group(2)) + float(wall.group(3)),
        "rss": int(rss.group(1)),
    }


def best(work, tag, stage):
    """Minimum wall over the repeats that exist, with that run's cpu and rss."""
    runs = [parse_time(p) for p in sorted(glob.glob(f"{work}/{tag}.{stage}.time.r*"))]
    runs = [r for r in runs if r]
    if not runs:
        one = parse_time(f"{work}/{tag}.{stage}.time")
        runs = [one] if one else []
    return min(runs, key=lambda r: r["wall"]) if runs else None


def find(path, pattern, cast=str):
    match = re.search(pattern, open(path).read())
    return cast(match.group(1).replace(",", "")) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    work = args.workdir

    rows = []
    for tag, records in SIZES:
        if not os.path.exists(f"{work}/{tag}.predict.log"):
            continue
        stages = {s: best(work, tag, s) for s in STAGES}
        if any(v is None for v in stages.values()):
            continue
        plog, clog = f"{work}/{tag}.predict.log", f"{work}/{tag}.cluster.log"
        row = {
            "records": records,
            "candidates": find(plog, r"Candidates\s+([\d,]+)", int),
            "edges": find(plog, r"Edges\s+([\d,]+)", int),
            "avoided_pct": find(plog, r"Comparisons avoided\s+[\d,]+ \(([\d.]+)%", float),
            "precision": find(clog, r"precision\s+([\d.]+)", float),
            "recall": find(clog, r"recall\s+([\d.]+)", float),
            "f1": find(clog, r"f1\s+([\d.]+)", float),
            "stages": stages,
            "wall": sum(v["wall"] for v in stages.values()),
            "cpu": sum(v["cpu"] for v in stages.values()),
            "rss": max(v["rss"] for v in stages.values()),
            "spill": 0,
        }
        rows.append(row)

    splink = []
    for tag, records in SIZES:
        path = f"{work}/splink_{tag}/timings.json"
        if not os.path.exists(path):
            continue
        report = json.load(open(path))
        outer = parse_time(f"{work}/splink_{tag}.time")
        report["records"] = records
        if outer:
            report["wall"] = outer["wall"]
            report["cpu"] = outer["cpu"]
            report["peak_rss_bytes"] = outer["rss"]
        splink.append(report)

    result = {"cpplink": rows, "splink": splink}
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)

    print(f"{'records':>10} {'candidates':>15} {'wall':>8} {'cpu':>9} "
          f"{'RSS MB':>7} {'cand/s':>12} {'F1':>7}")
    for r in rows:
        rate = r["candidates"] / r["stages"]["predict"]["wall"]
        print(f"{r['records']:>10,} {r['candidates']:>15,} {r['wall']:>8.1f} "
              f"{r['cpu']:>9.1f} {r['rss'] // 1048576:>7,} {rate:>12,.0f} "
              f"{r['f1']:>7.4f}")
    if splink:
        print()
        print(f"{'records':>10} {'splink wall':>12} {'RSS MB':>8} {'spill MB':>9} "
              f"{'edges':>10}")
        for s in splink:
            print(f"{s['records']:>10,} {s['wall']:>12.1f} "
                  f"{s['peak_rss_bytes'] // 1048576:>8,} "
                  f"{s['peak_spill_bytes'] // 1048576:>9,} {s['edges']:>10,}")


if __name__ == "__main__":
    main()
