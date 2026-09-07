# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Score either tool's cluster file against the planted duplicates.

Both sides are closed transitively before scoring, for the reason the parity
benchmark's scorer gives: a chain a-b-c asserts a-c whether or not that pair was
ever scored, and scoring a transitive partition against a non-transitive truth
list counts genuine duplicates as false positives.

The three counts are contingency sums rather than enumerations, so a
catastrophic merge is priced correctly without materialising its pairs.
"""
import argparse
import csv
import sys
from collections import Counter


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def choose2(n):
    return n * (n - 1) // 2


def labels_from_pairs(path):
    uf = UnionFind()
    with open(path) as handle:
        for row in csv.DictReader(handle):
            uf.union(row["id_a"], row["id_b"])
    return {x: uf.find(x) for x in uf.parent}


def labels_from_clusters(path):
    with open(path) as handle:
        reader = csv.DictReader(handle)
        key = "unique_id" if "unique_id" in reader.fieldnames else "id"
        return {row[key]: row["cluster_id"] for row in reader}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    truth = labels_from_pairs(args.truth)
    predicted = labels_from_clusters(args.clusters)

    truth_pairs = sum(choose2(n) for n in Counter(truth.values()).values())
    # Only records the truth file mentions can contribute a true positive, so
    # the asserted count is taken over the same restriction; every other record
    # is a singleton on both sides and contributes nothing to any of the three.
    seen = {r: predicted.get(r, r) for r in truth}
    asserted_all = sum(choose2(n) for n in Counter(predicted.values()).values())
    joint = Counter((truth[r], seen[r]) for r in truth)
    true_positives = sum(choose2(n) for n in joint.values())

    precision = true_positives / asserted_all if asserted_all else 0.0
    recall = true_positives / truth_pairs if truth_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{args.label or args.clusters}: truth {truth_pairs:,} asserted "
          f"{asserted_all:,} tp {true_positives:,} "
          f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f}")


if __name__ == "__main__":
    sys.exit(main())
