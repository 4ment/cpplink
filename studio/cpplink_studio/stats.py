# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Column statistics, value counts, and what two columns have in common.

Everything is a DuckDB query over the ``data`` view. The value counts of a
column are the one thing computed once and kept: the blocking closed forms,
the effective cardinality and the top values all read the same array, and at
20M rows it is the count per distinct value (a few million integers), never
the values themselves.

Two estimator choices are deliberate. The collision rate of a column is
``sum c(c-1) / (N(N-1))``, the chance two distinct random rows agree, and not
the plug-in ``sum p^2`` which reads ``1/n`` on a column of singletons. And the
pairwise numbers are computed on a sample, because 45 group-bys over 20M rows
is minutes for a page that should answer in seconds; the sample size is in
the result so the page can say so.
"""

import math
import re

import numpy as np

from .data import quoted, sql_string

# Values that usually mean "missing" rather than a value. Reported, never acted
# on without the user saying so: "0" is a genuine value in many columns.
SENTINELS = {"na", "n/a", "nan", "null", "none", "unknown", "unk", "-", "--", "?",
             "missing", "not available", "not known", "0", "9999", "99999",
             "1900-01-01", "1970-01-01", "9999-12-31", "0000-00-00"}

SAMPLE_ROWS = 100_000


def pairs_in(count):
    return count * (count - 1) // 2


def is_list_type(duckdb_type):
    return duckdb_type.upper().endswith("[]")


def value_expression(ds, column):
    """The expression a column's values are counted by: blanks are missing for
    text, a list contributes each of its elements, everything else is itself."""
    q = quoted(column)
    t = ds.duckdb_types[column].upper()
    if is_list_type(t):
        return f"unnest({q})", True
    if t in ("VARCHAR", "TEXT", "STRING"):
        return f"NULLIF(trim({q}), '')", False
    return q, False


def value_counts(ds, column):
    """Count per distinct non-null value, descending, as an int64 array, plus
    the number of null rows. For a list column the counts are per element and
    the nulls are rows holding no element."""
    expr, unnested = value_expression(ds, column)
    q = quoted(column)
    if unnested:
        sql = (f"SELECT count(*) AS c FROM (SELECT {expr} AS v FROM data) "
               f"WHERE v IS NOT NULL AND trim(v) <> '' GROUP BY v ORDER BY c DESC")
        nulls = ds.query(
            f"SELECT count(*) FROM data WHERE {q} IS NULL OR len({q}) = 0").fetchone()[0]
    else:
        sql = (f"SELECT count(*) AS c FROM data WHERE {expr} IS NOT NULL "
               f"GROUP BY {expr} ORDER BY c DESC")
        nulls = ds.query(f"SELECT count(*) FROM data WHERE {expr} IS NULL").fetchone()[0]
    counts = ds.query(sql).fetchnumpy()["c"].astype(np.int64)
    return counts, int(nulls)


def collision_rate(counts):
    """P(two distinct random non-null rows agree): sum c(c-1) / (N(N-1))."""
    n = int(counts.sum())
    if n < 2:
        return 0.0
    return float((counts * (counts - 1)).sum()) / (n * (n - 1))


def top_values(ds, column, k=15):
    expr, unnested = value_expression(ds, column)
    if unnested:
        sql = (f"SELECT CAST(v AS VARCHAR) AS v, count(*) AS c FROM "
               f"(SELECT {expr} AS v FROM data) WHERE v IS NOT NULL AND trim(v) <> '' "
               f"GROUP BY v ORDER BY c DESC, v LIMIT {int(k)}")
    else:
        sql = (f"SELECT CAST({expr} AS VARCHAR) AS v, count(*) AS c FROM data "
               f"WHERE {expr} IS NOT NULL GROUP BY {expr} ORDER BY c DESC, v "
               f"LIMIT {int(k)}")
    return ds.query(sql).fetchall()


def sentinel_values(ds, column):
    """Values that look like a missing marker, with their counts."""
    expr, unnested = value_expression(ds, column)
    listed = ", ".join(sql_string(s) for s in sorted(SENTINELS))
    source = f"(SELECT {expr} AS v FROM data)"
    sql = (f"SELECT CAST(v AS VARCHAR), count(*) AS c FROM {source} "
           f"WHERE lower(trim(CAST(v AS VARCHAR))) IN ({listed}) "
           f"GROUP BY v ORDER BY c DESC")
    return ds.query(sql).fetchall()


def column_stats(ds, column, counts=None, nulls=None):
    """One column's profile as a flat dict, from its value counts and a few
    type-specific queries."""
    if counts is None:
        counts, nulls = value_counts(ds, column)
    rows = ds.rows
    non_null = rows - nulls
    distinct = int(len(counts))
    u = collision_rate(counts)
    out = {
        "column": column,
        "duckdb_type": ds.duckdb_types[column],
        "rows": rows,
        "nulls": nulls,
        "null_share": nulls / rows if rows else 0.0,
        "distinct": distinct,
        "collision_rate": u,
        "effective_cardinality": (1.0 / u) if u > 0 else float(distinct),
        "bits": (-math.log2(u)) if u > 0 else float("nan"),
        "top_share": (int(counts[0]) / non_null) if distinct and non_null else 0.0,
        "largest_group": int(counts[0]) if distinct else 0,
        "singletons": int((counts == 1).sum()),
    }
    # A value is overrepresented when it holds far more rows than a value of
    # this column typically does; the threshold is loose on purpose, since the
    # point is to make "Smith" or "UNKNOWN" visible, not to decide anything.
    mean = non_null / distinct if distinct else 0.0
    out["overrepresented"] = [
        (v, c, c / non_null) for v, c in top_values(ds, column, 15)
        if mean > 0 and c > 20 * mean and c / non_null > 0.005
    ]
    out["sentinels"] = sentinel_values(ds, column)
    out.update(type_specific_stats(ds, column))
    return out


def type_specific_stats(ds, column):
    ds.ensure_sample(SAMPLE_ROWS)
    q = quoted(column)
    t = ds.duckdb_types[column].upper()
    if is_list_type(t):
        row = ds.query(f"SELECT avg(len({q})), max(len({q})) FROM data "
                       f"WHERE {q} IS NOT NULL").fetchone()
        return {"list_mean_length": row[0], "list_max_length": row[1]}
    if t in ("VARCHAR", "TEXT", "STRING"):
        expr = f"NULLIF(trim({q}), '')"
        row = ds.query(
            f"SELECT min(length(v)), avg(length(v)), max(length(v)), "
            f"avg(CASE WHEN v = lower(v) THEN 1 ELSE 0 END), "
            f"avg(CASE WHEN v = upper(v) THEN 1 ELSE 0 END), "
            f"avg(CASE WHEN regexp_matches(v, '[0-9]') THEN 1 ELSE 0 END) "
            f"FROM (SELECT {expr} AS v FROM sample) WHERE v IS NOT NULL").fetchone()
        return {"min_length": row[0], "mean_length": row[1], "max_length": row[2],
                "lower_share": row[3], "upper_share": row[4], "digit_share": row[5]}
    if t.startswith("DATE") or t.startswith("TIMESTAMP"):
        row = ds.query(
            f"SELECT min({q}), max({q}), "
            f"avg(CASE WHEN month({q}) = 1 AND day({q}) = 1 THEN 1 ELSE 0 END), "
            f"avg(CASE WHEN day({q}) = 1 THEN 1 ELSE 0 END) "
            f"FROM data WHERE {q} IS NOT NULL").fetchone()
        return {"min": str(row[0]), "max": str(row[1]), "jan_first_share": row[2],
                "day_first_share": row[3]}
    if t == "BOOLEAN":
        row = ds.query(f"SELECT avg(CASE WHEN {q} THEN 1.0 ELSE 0.0 END) FROM data "
                       f"WHERE {q} IS NOT NULL").fetchone()
        return {"true_share": row[0]}
    if any(t.startswith(p) for p in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "INT",
                                     "BIGINT", "SMALLINT", "TINYINT", "HUGEINT",
                                     "UINT", "UBIGINT", "USMALLINT", "UTINYINT")):
        row = ds.query(f"SELECT min({q}), max({q}), avg({q}) FROM data "
                       f"WHERE {q} IS NOT NULL").fetchone()
        return {"min": row[0], "max": row[1], "mean": row[2]}
    return {}


# --- content probes, for guessing what a column holds ------------------------

EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE = r"^\+?[0-9 ()./-]{6,20}$"
POSTCODE = r"^[A-Za-z0-9][A-Za-z0-9 -]{2,9}$"
NAME = r"^[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ' .-]{0,40}$"
GENDER = ("m", "f", "male", "female", "man", "woman", "other", "u", "x", "nb")


def probe_content(ds, column):
    """Fractions of a sample's non-null values that look like an email, a
    phone number, a postcode, a date, a personal name, a gender code, and
    a coordinate. Regexes on a sample: a hint for the role guesser, never a
    statistic the page should show as fact."""
    q = quoted(column)
    t = ds.duckdb_types[column].upper()
    if is_list_type(t):
        return {"kind": "list"}
    text = f"NULLIF(trim(CAST({q} AS VARCHAR)), '')"
    genders = ", ".join(sql_string(g) for g in GENDER)
    sql = (
        f"SELECT count(*), "
        f"avg(CASE WHEN regexp_matches(v, {sql_string(EMAIL)}) THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN regexp_matches(v, {sql_string(PHONE)}) "
        f"  AND length(regexp_replace(v, '[^0-9]', '', 'g')) BETWEEN 6 AND 15 "
        f"  THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN regexp_matches(v, {sql_string(POSTCODE)}) "
        f"  AND regexp_matches(v, '[0-9]') THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN TRY_CAST(v AS DATE) IS NOT NULL THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN regexp_matches(v, {sql_string(NAME)}) THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN lower(v) IN ({genders}) THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN TRY_CAST(v AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN TRY_CAST(v AS DOUBLE) BETWEEN -90 AND 90 THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN TRY_CAST(v AS DOUBLE) BETWEEN -180 AND 180 THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN regexp_matches(v, '^[0-9]+$') THEN 1 ELSE 0 END), "
        f"avg(length(v)) "
        f"FROM (SELECT {text} AS v FROM sample) WHERE v IS NOT NULL")
    ds.ensure_sample(SAMPLE_ROWS)
    row = ds.query(sql).fetchone()
    keys = ["sampled", "email", "phone", "postcode", "date", "name", "gender",
            "numeric", "lat_range", "lon_range", "digits", "mean_length"]
    out = dict(zip(keys, row))
    out["kind"] = "scalar"
    return out


# --- what two columns share ---------------------------------------------------

def sample_table(ds, rows=1_000_000):
    """Materialise a sample once for the pairwise pass; the whole file when it
    is no larger than the sample. Returns the row count of the sample."""
    if ds.rows <= rows:
        ds.query("CREATE OR REPLACE TABLE pair_sample AS SELECT * FROM data")
        return ds.rows
    ds.query(f"CREATE OR REPLACE TABLE pair_sample AS SELECT * FROM data "
             f"USING SAMPLE reservoir({int(rows)} ROWS) REPEATABLE (1)")
    return ds.query("SELECT count(*) FROM pair_sample").fetchone()[0]


def pair_stats(ds, a, b):
    """What column ``a`` says about column ``b`` on the pair sample.

    ``containment_a_in_b``: share of rows where a's text is inside b's, the
    shape ``first_name`` inside ``first_and_surname`` takes.
    ``determination_a_to_b``: share of rows whose a value maps to its commonest
    b value, against the ``baseline`` share of b's commonest value overall;
    refused (None) when a is near-unique, since a key determines everything.
    ``overlap_bits``: log2 of the collision rate of (a, b) over the product of
    the two rates, the u-side overlap a matching weight double counts; marked
    unreliable when the independent expectation is under fifty collisions,
    because then the observed collisions are the file's own duplicates.
    """
    ea, la = value_expression(ds, a)
    eb, lb = value_expression(ds, b)
    if la or lb:
        return None
    both = (f"(SELECT CAST({ea} AS VARCHAR) AS x, CAST({eb} AS VARCHAR) AS y "
            f"FROM pair_sample WHERE {ea} IS NOT NULL AND {eb} IS NOT NULL)")
    n = ds.query(f"SELECT count(*) FROM {both}").fetchone()[0]
    out = {"a": a, "b": b, "both_non_null": n}
    if n < 100:
        return out
    row = ds.query(
        f"SELECT avg(CASE WHEN contains(lower(y), lower(x)) THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN contains(lower(x), lower(y)) THEN 1 ELSE 0 END) FROM {both}"
    ).fetchone()
    out["containment_a_in_b"], out["containment_b_in_a"] = row
    da = ds.query(f"SELECT count(DISTINCT x), count(DISTINCT y) FROM {both}").fetchone()
    out["distinct_a"], out["distinct_b"] = da
    joint = (f"(SELECT x, y, count(*) AS c FROM {both} GROUP BY x, y)")
    if da[0] < 0.9 * n:
        det = ds.query(f"SELECT sum(m) FROM (SELECT x, max(c) AS m FROM {joint} "
                       f"GROUP BY x)").fetchone()[0]
        base = ds.query(f"SELECT max(c) FROM (SELECT y, count(*) AS c FROM {both} "
                        f"GROUP BY y)").fetchone()[0]
        out["determination_a_to_b"] = det / n
        out["baseline_b"] = base / n
    if da[1] < 0.9 * n:
        det = ds.query(f"SELECT sum(m) FROM (SELECT y, max(c) AS m FROM {joint} "
                       f"GROUP BY y)").fetchone()[0]
        base = ds.query(f"SELECT max(c) FROM (SELECT x, count(*) AS c FROM {both} "
                        f"GROUP BY x)").fetchone()[0]
        out["determination_b_to_a"] = det / n
        out["baseline_a"] = base / n
    # Collision rates on the same rows, so the three share a denominator.
    ua = ds.query(f"SELECT sum(c * (c - 1)) FROM (SELECT x, count(*) AS c FROM {both} "
                  f"GROUP BY x)").fetchone()[0] or 0
    ub = ds.query(f"SELECT sum(c * (c - 1)) FROM (SELECT y, count(*) AS c FROM {both} "
                  f"GROUP BY y)").fetchone()[0] or 0
    uab = ds.query(f"SELECT sum(c * (c - 1)) FROM {joint}").fetchone()[0] or 0
    denom = n * (n - 1)
    ua, ub, uab = ua / denom, ub / denom, uab / denom
    out["u_a"], out["u_b"], out["u_ab"] = ua, ub, uab
    expected = ua * ub * denom / 2
    out["expected_joint_collisions"] = expected
    out["observed_joint_collisions"] = uab * denom / 2
    if ua > 0 and ub > 0 and uab > 0:
        out["overlap_bits"] = math.log2(uab / (ua * ub))
        out["overlap_reliable"] = expected >= 50  # refined by flag_reliability
    # Null co-occurrence, over the whole sample rather than the non-null rows.
    qa, qb = quoted(a), quoted(b)
    row = ds.query(
        f"SELECT avg(CASE WHEN {qb} IS NULL THEN 1 ELSE 0 END), "
        f"avg(CASE WHEN {qb} IS NULL THEN 1 ELSE 0 END) FILTER (WHERE {qa} IS NULL) "
        f"FROM pair_sample").fetchone()
    out["null_b"], out["null_b_given_null_a"] = row
    return out


def flag_reliability(pairs, margin=5.0):
    """Decide which overlaps are dependence and which are the file's duplicates.

    On a file holding duplicates, two near-unique columns collide together
    almost only on duplicate pairs, so the joint reads as strong dependence
    between columns that are independent. The excess of observed over expected
    joint collisions on any pair is a lower bound on the number of duplicate
    pairs agreeing on both, and the largest excess over every pair measured is
    the floor the file's duplicates put under every joint. A joint is trusted
    only where its independent expectation exceeds that floor by ``margin``,
    so duplicates can move it by at most ``log2(1 + 1/margin)`` bits.
    Returns the floor; sets ``overlap_reliable`` on every pair in place.
    """
    floor = 0.0
    for p in pairs.values():
        if p is None or "observed_joint_collisions" not in p:
            continue
        excess = p["observed_joint_collisions"] - p["expected_joint_collisions"]
        floor = max(floor, excess)
    for p in pairs.values():
        if p is None or "overlap_bits" not in p:
            continue
        expected = p["expected_joint_collisions"]
        p["overlap_reliable"] = expected >= 50 and expected >= margin * floor
        p["duplicate_floor"] = floor
    return floor


def normalise_name(name):
    """A column name reduced to letters, for the synonym table."""
    return re.sub(r"[^a-z]", "", name.lower())
