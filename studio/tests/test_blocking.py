# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The closed forms agree with `explain-blocking` source by source, including
a derived column reproduced in SQL and a window; the curves are monotone; the
drafted plan fits its budget."""

import json

import numpy as np

from cpplink_studio import binary, blocking, schema


def priced_by_binary(schema_dict, parquet):
    report, _, _ = binary.explain_blocking(json.dumps(schema_dict), [parquet])
    return report


def test_plan_matches_binary(dataset, sample, sample_schema):
    types = {c["name"]: c.get("type", "string") for c in sample_schema["columns"]}
    pricer = blocking.Pricer(dataset, types)
    ours = blocking.price_plan(pricer, sample_schema["blocking"], sample_schema["columns"],
                               dataset.rows)
    theirs = priced_by_binary(sample_schema, sample["parquet"])
    assert theirs["records"] == dataset.rows
    assert theirs["pair_space"] == ours["pair_space"]
    assert ours["unpriced"] == 0
    for mine, its in zip(ours["sources"], theirs["sources"]):
        assert mine["candidates"] == its["candidate_pairs"], its["name"]
        if its["type"] != "sorted_neighbourhood":
            assert mine["largest_group"] == its["largest_group"], its["name"]
    assert ours["candidate_sum"] == theirs["candidate_sum"]


def test_derived_and_window_match_binary(dataset, sample, sample_schema):
    plan = [{"type": "exact_value", "column": "email_username"},
            {"type": "rare_value", "column": "first_name", "max_frequency": 30},
            {"type": "sorted_neighbourhood", "column": "email", "window": 5},
            {"type": "exact_value", "column": "gender"},
            {"type": "exact_value", "column": "dob"}]
    schema_dict = dict(sample_schema, blocking=plan)
    types = {c["name"]: c.get("type", "string") for c in sample_schema["columns"]}
    pricer = blocking.Pricer(dataset, types)
    ours = blocking.price_plan(pricer, plan, sample_schema["columns"], dataset.rows)
    theirs = priced_by_binary(schema_dict, sample["parquet"])
    assert ours["unpriced"] == 0
    for mine, its in zip(ours["sources"], theirs["sources"]):
        assert mine["candidates"] == its["candidate_pairs"], its["name"]


def test_all_pairs_is_the_pair_space(dataset, sample, sample_schema):
    plan = [{"type": "all_pairs"}]
    types = {c["name"]: c.get("type", "string") for c in sample_schema["columns"]}
    pricer = blocking.Pricer(dataset, types)
    ours = blocking.price_plan(pricer, plan, sample_schema["columns"], dataset.rows)
    theirs = priced_by_binary(dict(sample_schema, blocking=plan), sample["parquet"])
    assert ours["sources"][0]["candidates"] == theirs["sources"][0]["candidate_pairs"]
    assert ours["sources"][0]["candidates"] == theirs["pair_space"]


def test_curves_are_monotone():
    counts = np.array([1, 1, 2, 3, 5, 8, 13, 21, 400], dtype=np.int64)
    curve = blocking.rare_curve(counts, [1, 2, 4, 8, 16, 32, 1000])
    cands = [p["candidates"] for p in curve]
    assert cands == sorted(cands)
    assert curve[-1]["candidates"] == blocking.exact_pairs(counts)
    assert curve[-1]["coverage"] == 1.0
    assert curve[0]["candidates"] == 0
    for cap, point in zip([1, 2, 4, 8, 16, 32, 1000], curve):
        assert point["candidates"] == blocking.rare_pairs(counts, cap)
    assert blocking.window_pairs(10, 3) == 7 * 3 + 3
    assert blocking.window_pairs(3, 10) == 3  # window clamped to count - 1
    hist = blocking.group_size_histogram(counts)
    assert sum(b["values"] for b in hist) == len(counts)
    assert sum(b["pairs"] for b in hist) == blocking.exact_pairs(counts)


def test_draft_plan_fits_budget(dataset, sample_schema):
    types = {c["name"]: c.get("type", "string") for c in sample_schema["columns"]
             if "derive" not in c}
    roles = {"first_name": "first_name", "last_name": "surname", "dob": "dob",
             "email": "email", "phone": "phone", "postcode": "postcode",
             "gender": "gender", "address_tokens": "address"}
    # A list column takes the address role but is never blocked on.
    pricer = blocking.Pricer(dataset, types)
    derived = [{"name": "email_username", "from": "email", "transform": "email_username"}]
    columns = sample_schema["columns"]
    plan, reasons = blocking.draft_plan(pricer, roles, types, columns, dataset.rows,
                                        budget_per_row=50, derived=derived)
    assert plan and len(reasons) == len(plan)
    priced = blocking.price_plan(pricer, plan, columns, dataset.rows)
    assert priced["unpriced"] == 0
    assert priced["candidate_sum"] <= 50 * dataset.rows * 1.5
    assert any(s["column"] == "email_username" for s in plan)
    assert all(s["column"] != "address_tokens" for s in plan)
    assert all(s["type"] != "minhash" for s in plan)
    # The drafted plan is a schema the binary accepts.
    draft = schema.from_schema(dict(sample_schema, blocking=plan), dataset)
    assert schema.problems(draft) == []
