# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Pairwise precision and recall of a linkage, at every threshold, one scorer.

A link between two tables is a set of cross pairs, and here the truth is one
pair per origin record: origin i paid destination i. So the score is over
predicted pairs directly, with no clustering and no closure -- precision is the
share of pairs at or above the threshold that are true, recall the share of the
45,326 true pairs found. Both tools write the same three columns and this reads
them the same way.

The best F1 over the grid is what the summary quotes, and the threshold it sits
at is printed beside it, because the two tools' probabilities are not on the
same scale where their estimates differ.
"""

import argparse
import json

import pandas as pd


def score(predictions_path, truth_pairs, thresholds):
    frame = pd.read_csv(predictions_path, dtype={"origin_id": str, "destination_id": str})
    hit = frame["origin_id"] == frame["destination_id"]
    rows = []
    for threshold in thresholds:
        above = frame["match_probability"] >= threshold
        predicted = int(above.sum())
        true = int((above & hit).sum())
        precision = true / predicted if predicted else 0.0
        recall = true / truth_pairs if truth_pairs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "predicted": predicted,
                "true": true,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--truth-pairs", type=int, default=45326)
    parser.add_argument("--thresholds", default="0.001,0.01,0.1,0.5,0.9,0.99,0.999")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(json.dumps(score(args.predictions, args.truth_pairs, thresholds), indent=2))


if __name__ == "__main__":
    main()
