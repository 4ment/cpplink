# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Write the parity schema both tools compile from.

Only levels splink implements identically are used: exact match, Levenshtein at
a distance, Jaro-Winkler at a similarity. That drops the date, geo and list
levels the sample schema carries, which makes the model slightly worse in both
tools, equally. Blocking is exact agreement on single columns, which both tools
express the same way.
"""
import json
import sys


def comparison(column, levels, term_frequency=True):
    packed = [{"type": "null"}, {"type": "exact"}]
    for kind, threshold in levels:
        name = "jaro_winkler" if kind == "jw" else "levenshtein"
        packed.append({"type": name, "threshold": threshold})
    packed.append({"type": "else"})
    return {
        "name": column,
        "columns": [column],
        "term_frequency": term_frequency,
        "levels": packed,
    }


SCHEMA = {
    "unique_id": "id",
    "columns": [
        {"name": "first_name", "type": "string"},
        {"name": "last_name", "type": "string"},
        {"name": "dob", "type": "date"},
        {"name": "email", "type": "string"},
        {"name": "phone", "type": "string"},
        {"name": "postcode", "type": "string"},
    ],
    "comparisons": [
        comparison("last_name", [("jw", 0.92), ("jw", 0.85)]),
        comparison("first_name", [("lev", 1), ("jw", 0.88)]),
        comparison("email", [("jw", 0.93)]),
        comparison("phone", [("lev", 2)]),
        comparison("dob", []),
        comparison("postcode", []),
    ],
    "blocking": [
        {"type": "exact_value", "column": c}
        for c in ["email", "phone", "dob", "postcode", "last_name"]
    ],
}

if __name__ == "__main__":
    with open(sys.argv[1], "w") as handle:
        json.dump(SCHEMA, handle, indent=2)
