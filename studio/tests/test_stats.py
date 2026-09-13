# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT

import math

import numpy as np

from cpplink_studio import stats


def test_collision_rate_is_the_pair_form():
    counts = np.array([3, 1, 1], dtype=np.int64)
    # 5 rows, 10 pairs, 3 of them agree.
    assert math.isclose(stats.collision_rate(counts), 3 / 10)
    assert stats.collision_rate(np.array([1, 1, 1], dtype=np.int64)) == 0.0


def test_column_stats_shape(dataset):
    st = stats.column_stats(dataset, "last_name")
    assert st["rows"] == dataset.rows
    assert st["nulls"] + int(sum(c for _, c in stats.top_values(dataset, "last_name", 10**9))) \
        == dataset.rows
    # The collision form can read above the distinct count on a column of
    # mostly singletons; it is 1/u, not a count of values.
    assert st["effective_cardinality"] >= 1.0
    assert st["bits"] == -math.log2(st["collision_rate"])
    assert "mean_length" in st
    st = stats.column_stats(dataset, "address_tokens")
    assert "list_mean_length" in st
    st = stats.column_stats(dataset, "dob")
    assert st["min"] < st["max"]


def test_pair_stats_sees_the_duplicates(dataset):
    stats.sample_table(dataset)
    ps = stats.pair_stats(dataset, "email", "phone")
    # Two near-unique columns collide together only on the planted duplicates,
    # so the joint collisions dwarf the independent expectation and the overlap
    # is marked unreliable rather than reported as dependence.
    assert ps["observed_joint_collisions"] > 10 * ps["expected_joint_collisions"]
    assert ps["overlap_reliable"] is False
    assert ps.get("determination_a_to_b") is None  # near-unique determinant refused
    ps = stats.pair_stats(dataset, "first_name", "gender")
    assert ps["determination_a_to_b"] is not None
    assert ps["baseline_b"] > 0.4


def test_duplicate_floor_silences_the_planted_pairs(dataset):
    """Every column of gen-sample is drawn independently, so the only joint
    collisions beyond chance are the planted duplicates: with the floor in
    place no pair may read as a reliable dependence, except through the email,
    which is built from the names and really does contain them."""
    stats.sample_table(dataset)
    columns = ["first_name", "last_name", "dob", "phone", "postcode"]
    pairs = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pairs[(a, b)] = stats.pair_stats(dataset, a, b)
    floor = stats.flag_reliability(pairs)
    assert floor > 100
    assert not any(p.get("overlap_reliable") for p in pairs.values())
