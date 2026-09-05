# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""One scorer, used for both tools, so the numbers are commensurable.

Quality is pairwise precision and recall over the *transitive closure* of the
partition: a chain a-b-c asserts a-c whether or not that pair was ever scored.
That is a stricter number than edge precision and is the one that matters, and
it is the same definition cpplink's `cluster --truth` reports -- but computed
here, once, from the cluster files, so that neither tool's own accounting is
taken on trust.

The three counts are contingency sums, not enumerations: with n_e records of
true entity e, n_c records in predicted cluster c and n_ec in both,

    truth pairs     = sum_e  C(n_e, 2)
    asserted pairs  = sum_c  C(n_c, 2)
    true positives  = sum_ec C(n_ec, 2)

so a catastrophic merge is priced correctly without materialising its pairs.
"""

import argparse
import json

import pandas as pd


def _choose2(counts):
    return int((counts * (counts - 1) // 2).sum())


def score(entities_path, clusters_path):
    truth = pd.read_csv(entities_path, dtype=str)
    predicted = pd.read_csv(clusters_path, dtype=str)
    if "cluster_id" not in predicted.columns or "unique_id" not in predicted.columns:
        raise SystemExit(f"{clusters_path}: want unique_id and cluster_id columns")
    predicted = predicted[["unique_id", "cluster_id"]].drop_duplicates("unique_id")

    frame = truth.merge(predicted, on="unique_id", how="left")
    # A record no tool put in a cluster is a singleton, and singletons must be
    # distinct from each other or they would count as one giant cluster.
    missing = frame["cluster_id"].isna()
    frame.loc[missing, "cluster_id"] = "\x00singleton\x00" + frame.loc[
        missing, "unique_id"
    ].astype(str)

    truth_pairs = _choose2(frame.groupby("entity").size())
    asserted = _choose2(frame.groupby("cluster_id").size())
    hits = _choose2(frame.groupby(["entity", "cluster_id"]).size())

    precision = hits / asserted if asserted else 0.0
    recall = hits / truth_pairs if truth_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sizes = frame.groupby("cluster_id").size()
    return {
        "records": int(len(frame)),
        "truth_pairs": truth_pairs,
        "asserted_pairs": asserted,
        "true_positives": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clusters": int((sizes > 1).sum()),
        "singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entities")
    parser.add_argument("clusters")
    args = parser.parse_args()
    print(json.dumps(score(args.entities, args.clusters), indent=2))


if __name__ == "__main__":
    main()
