# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Price a blocking plan from value counts alone, and draft one under a budget.

The candidate count of an exact-value or rare-value source is closed form in
the term frequencies, ``sum c(c-1)/2`` over the values under the cap, and a
sorted-neighbourhood window is closed form in the number of non-null rows.
These are the same formulas ``BlockingPlan::CountPairs`` uses, so the page
can move a cap on a slider and watch the count move without running anything;
``tests/test_blocking.py`` holds the two implementations to the same numbers.

What is not closed form is priced by the binary: MinHash bands, the
deduplicated union of a plan, and any source on a derived column whose
transform is not reproduced here (``soundex``, ``normalize``,
``sorted_tokens``). The email and date-part transforms are exact in SQL and
are reproduced, so a source on ``email_username`` prices live.

Two measured costs turn a candidate count into a wall-clock estimate, both
single-threaded on the synthetic sample: about 207 ns per candidate with the
score ceiling on, about 4,090 ns without it.
"""

import math

import numpy as np

from .data import quoted
from .stats import value_counts

NS_PER_CANDIDATE_CEILING = 207.0
NS_PER_CANDIDATE_FULL = 4090.0
NS_PER_CANDIDATE_ENUMERATE = 9.0

SOURCE_TYPES = ["exact_value", "rare_value", "sorted_neighbourhood", "minhash",
                "all_pairs"]

# Transforms reproduced in SQL, exactly as derive.cpp computes them. A derived
# column declared with any other transform is priced by the binary only.
SQL_TRANSFORMS = {
    "email_username": lambda v: f"CASE WHEN strpos({v}, '@') > 0 THEN "
                                f"substr({v}, 1, strpos({v}, '@') - 1) ELSE {v} END",
    "email_domain": lambda v: f"CASE WHEN strpos({v}, '@') > 0 THEN "
                              f"substr({v}, length({v}) - strpos(reverse({v}), '@') + 2) "
                              f"ELSE NULL END",
    "year": lambda v: f"CAST(year({v}) AS VARCHAR)",
    "month": lambda v: f"CAST(month({v}) AS VARCHAR)",
    "day": lambda v: f"CAST(day({v}) AS VARCHAR)",
    "year_month": lambda v: f"strftime({v}, '%Y-%m')",
}


def pairs_in(count):
    return count * (count - 1) // 2


def exact_pairs(counts):
    return int((counts * (counts - 1) // 2).sum())


def rare_pairs(counts, cap):
    if cap == 0:
        return exact_pairs(counts)
    kept = counts[counts <= cap]
    return int((kept * (kept - 1) // 2).sum())


def window_pairs(non_null_rows, window):
    """(count - w) * w + w(w-1)/2, as CountPooledPairs computes it."""
    count = int(non_null_rows)
    if count < 2:
        return 0
    w = min(int(window), count - 1)
    return (count - w) * w + pairs_in(w)


def rare_curve(counts, caps):
    """Candidate pairs and the share of rows the source can see, per cap.

    Sorted ascending once, then each cap is a prefix sum: the whole curve is
    a few array operations however many caps are asked for.
    """
    asc = np.sort(counts)
    pairs = np.cumsum(asc * (asc - 1) // 2)
    rows = np.cumsum(asc)
    total = int(asc.sum())
    idx = np.searchsorted(asc, np.asarray(caps), side="right")
    cand = np.where(idx > 0, pairs[np.maximum(idx - 1, 0)], 0)
    seen = np.where(idx > 0, rows[np.maximum(idx - 1, 0)], 0)
    return [{"cap": int(c), "candidates": int(p),
             "coverage": (int(r) / total) if total else 0.0}
            for c, p, r in zip(caps, cand, seen)]


def window_curve(non_null_rows, windows):
    return [{"window": int(w), "candidates": window_pairs(non_null_rows, w)}
            for w in windows]


def group_size_histogram(counts):
    """Values, rows and pairs per power-of-two bucket of group size, so a cap
    can be read off: everything in the buckets above it is what the cap cuts."""
    if len(counts) == 0:
        return []
    top = int(counts.max())
    edges = [1]
    while edges[-1] < top:
        edges.append(edges[-1] * 2)
    edges.append(edges[-1] * 2)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (counts >= lo) & (counts < hi)
        if not mask.any():
            continue
        bucket = counts[mask]
        out.append({"low": lo, "high": hi - 1, "values": int(mask.sum()),
                    "rows": int(bucket.sum()), "pairs": int((bucket * (bucket - 1) // 2).sum())})
    return out


def default_caps(counts):
    """A cap grid that brackets the knee: geometric, ratio about 1.4, from 2
    up to the largest group, which is always the last point."""
    top = int(counts.max()) if len(counts) else 1
    caps = {2, top}
    cap = 2.0
    while cap < top:
        cap *= 1.4142
        caps.add(int(round(cap)))
    return sorted(c for c in caps if 2 <= c <= top) or [2]


class Pricer:
    """Value counts per (column, transform), computed once and kept."""

    def __init__(self, ds, types):
        self.ds = ds
        self.types = types
        self._counts = {}

    def counts(self, column, derive=None):
        """(counts, nulls) for a file column, or for a derived column given as
        (source column, transform). None when the transform is not reproduced."""
        key = (column, derive)
        if key in self._counts:
            return self._counts[key]
        if derive is None:
            result = value_counts(self.ds, column)
        else:
            result = self._derived_counts(column, derive)
        self._counts[key] = result
        return result

    def _derived_counts(self, source, transform):
        if transform not in SQL_TRANSFORMS:
            return None
        q = quoted(source)
        base = f"NULLIF(trim(CAST({q} AS VARCHAR)), '')" \
            if self.types.get(source) == "string" else q
        expr = SQL_TRANSFORMS[transform](base)
        sql = (f"SELECT count(*) AS c FROM (SELECT NULLIF({expr}, '') AS v FROM data "
               f"WHERE {q} IS NOT NULL) WHERE v IS NOT NULL GROUP BY v ORDER BY c DESC")
        counts = self.ds.query(sql).fetchnumpy()["c"].astype(np.int64)
        nulls = self.ds.rows - int(counts.sum())
        return counts, nulls


def resolve_column(name, columns):
    """(file column, transform or None) for a schema column name. A derived
    column with a chain longer than one, or a transform not in SQL, gives None."""
    for spec in columns:
        if spec["name"] != name:
            continue
        derive = spec.get("derive")
        if derive is None:
            return name, None
        transforms = derive.get("transform")
        if isinstance(transforms, str):
            transforms = [transforms]
        if len(transforms) != 1 or transforms[0] not in SQL_TRANSFORMS:
            return None
        return derive["from"], transforms[0]
    return None


def price_source(pricer, source, columns, rows):
    """The candidate count of one source, or None when only the binary can say.
    Returns a dict with ``candidates``, ``coverage``, ``largest_group`` and
    ``exact`` (False when the number is an estimate)."""
    kind = source["type"]
    if kind == "all_pairs":
        return {"candidates": pairs_in(rows), "coverage": 1.0, "largest_group": rows,
                "exact": True}
    if kind == "minhash":
        return None
    resolved = resolve_column(source["column"], columns)
    if resolved is None:
        return None
    got = pricer.counts(*resolved)
    if got is None:
        return None
    counts, nulls = got
    non_null = rows - nulls
    # A list column's counts are per element, so its coverage is the share of
    # rows holding any element, which is all the counts can say.
    if kind == "exact_value":
        return {"candidates": exact_pairs(counts),
                "coverage": min(1.0, non_null / rows) if rows else 0,
                "largest_group": int(counts.max()) if len(counts) else 0, "exact": True}
    if kind == "rare_value":
        cap = int(source.get("max_frequency", 100))
        kept = counts[counts <= cap] if cap else counts
        return {"candidates": rare_pairs(counts, cap),
                "coverage": min(1.0, int(kept.sum()) / rows) if rows else 0,
                "largest_group": int(kept.max()) if len(kept) else 0, "exact": True}
    if kind == "sorted_neighbourhood":
        # The window walks rows, not values, and a list column's count is per
        # element, so only a scalar column's window is priced here.
        return {"candidates": window_pairs(non_null, int(source.get("window", 8))),
                "coverage": non_null / rows if rows else 0,
                "largest_group": int(source.get("window", 8)) + 1, "exact": True}
    return None


def price_plan(pricer, blocking, columns, rows):
    """Per source pricing and the sum, which bounds the union from above."""
    priced = []
    total = 0
    unpriced = 0
    for source in blocking:
        p = price_source(pricer, source, columns, rows)
        if p is None:
            unpriced += 1
        else:
            total += p["candidates"]
        priced.append(p)
    return {"sources": priced, "candidate_sum": total, "unpriced": unpriced,
            "pair_space": pairs_in(rows)}


def wall_estimate(candidates, threads=1):
    """Seconds to score this many candidates, with and without the ceiling."""
    t = max(1, int(threads))
    return {"ceiling_s": candidates * NS_PER_CANDIDATE_CEILING / 1e9 / t,
            "full_s": candidates * NS_PER_CANDIDATE_FULL / 1e9 / t,
            "enumerate_s": candidates * NS_PER_CANDIDATE_ENUMERATE / 1e9 / t}


def cap_for_budget(counts, budget):
    """The largest cap on the default grid whose candidate count fits the budget."""
    best = 2
    for cap in default_caps(counts):
        if rare_pairs(counts, cap) <= budget:
            best = cap
        else:
            break
    return best


def draft_plan(pricer, roles, types, columns, rows, budget_per_row=500, derived=()):
    """A default plan under a candidate budget of ``budget_per_row * rows``.

    Exact-value sources go on the identifier-like columns whose count fits;
    rare-value sources go on the name-like columns with the cap solved from
    what is left of the budget, shared equally; one sorted neighbourhood on
    the strongest surname column catches the typos in the key. A derived
    email username is an exact-value source, as the templates make it.
    MinHash is never proposed: single-column MinHash is dominated on every
    dataset the project measured.
    """
    from .templates import blocking_for
    budget = budget_per_row * rows
    plan = []
    reasons = []
    wanted = []
    for name, role in roles.items():
        if name not in types:
            continue
        source = blocking_for(name, role, types[name])
        if source is not None:
            wanted.append((name, role, source))
    for d in derived:
        if d["transform"] == "email_username":
            wanted.append((d["name"], "email", {"type": "exact_value", "column": d["name"]}))
    spent = 0
    rare = []
    for name, role, source in wanted:
        if source["type"] != "exact_value":
            rare.append((name, role, source))
            continue
        resolved = resolve_column(name, columns)
        got = pricer.counts(*resolved) if resolved else None
        if got is None:
            continue
        cost = exact_pairs(got[0])
        if cost <= budget * 0.5:
            plan.append(source)
            spent += cost
            reasons.append(f"{name}: exact agreement costs {cost:,} candidates")
        else:
            cap = cap_for_budget(got[0], budget * 0.25)
            capped = {"type": "rare_value", "column": name, "max_frequency": cap}
            plan.append(capped)
            spent += rare_pairs(got[0], cap)
            reasons.append(f"{name}: exact agreement would cost {cost:,}, capped at {cap}")
    remaining = max(budget - spent, 0)
    share = remaining / max(len(rare), 1)
    for name, role, source in rare:
        got = pricer.counts(name)
        if got is None:
            continue
        cap = cap_for_budget(got[0], share)
        if cap >= int(got[0].max()):
            plan.append({"type": "exact_value", "column": name})
            reasons.append(f"{name}: every group fits {share:,.0f} candidates, "
                           f"so exact agreement")
        else:
            plan.append(dict(source, max_frequency=cap))
            reasons.append(f"{name}: rare-value cap {cap} fits {share:,.0f} candidates")
        spent += rare_pairs(got[0], cap)
    surname = next((n for n, r in roles.items() if r == "surname" and n in types), None)
    if surname is not None and rows > 0:
        w = 8
        if window_pairs(rows, w) <= max(budget - spent, 0):
            plan.append({"type": "sorted_neighbourhood", "column": surname, "window": w})
            reasons.append(f"{surname}: window {w} for typos in the key")
    return plan, reasons


def format_seconds(s):
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def bits(u):
    return -math.log2(u) if u > 0 else float("inf")
