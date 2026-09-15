# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The transactions linkage benchmark: one description, both tools compile from it.

splink's documentation links two fake banking tables, `transactions_origin` and
`transactions_destination` from `splink_datasets`, 45,326 rows each, where
origin row i is the payment that destination row i received: money shows up
with a delay of days, the amount differs by fees and exchange, and the memo is
truncated or altered. The two tables share their `unique_id` space, so a record
is named by its dataset and its id, which is the case cpplink's link path is
built for.

Three configurations:

  demo     splink exactly as its documentation runs it: the six SQL blocking
           rules (one of them `block_on("unique_id")`, which the demo itself
           calls a cheat), directed date levels, EM on memo then amount. The
           reference number. cpplink runs the nearest thing it can express --
           the same levels except a symmetric date window, and the same rules
           as key equalities on precomputed columns, which for two of them is a
           superset of the demo's candidates because an amount-ratio conjunct
           has no key form.
  matched  both tools on identical candidate sets: the demo's rules made into
           key equalities on precomputed columns, without the cheat, verified
           to the pair by counting each rule in both tools. The parity test.
  native   cpplink's matched plan plus its automatic sources, cpplink only.
  directed splink only: the matched blocking with the demo's directed date
           levels, which separates what the direction of the date is worth
           from what the demo's tighter blocking is worth.

The blocking keys are computed by duckdb from the demo's own SQL expressions
(`KEYS`), once, into the parquet both tools read, so a key means the same thing
in both by construction. Two keys are asymmetric -- the demo shifts the origin
date by 15 days and the origin amount by a dollar -- and are computed from a
different expression per file, which is what the rule says and what a link
between two files allows.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
DATA = os.path.join(BENCH, "data", "transactions")
SCHEMAS = os.path.join(BENCH, "schemas")
RESULTS = os.path.join(BENCH, "results", "transactions")

ORIGIN = os.path.join(DATA, "origin.parquet")
DESTINATION = os.path.join(DATA, "destination.parquet")
TRUTH = os.path.join(DATA, "truth.csv")
ROWS = 45326

# The demo's lambda: one match per origin record, over N * N cross pairs.
LAMBDA = 1.0 / ROWS

# Blocking keys, as duckdb SQL over one table, in the order the demo lists its
# rules. Where the demo's rule reads the two sides differently the origin and
# destination expressions differ; where a rule carries a conjunct that is not
# an equality (the amount ratio in the two date rules) the key drops it and
# the demo track notes the superset.
KEYS = [
    ("k_month_memo3",
     "strftime(transaction_date, '%Y%m') || '|' || substr(memo, 1, 3)",
     "strftime(transaction_date, '%Y%m') || '|' || substr(memo, 1, 3)"),
    ("k_month15_memo3",
     "strftime(transaction_date + 15, '%Y%m') || '|' || substr(memo, 1, 3)",
     "strftime(transaction_date, '%Y%m') || '|' || substr(memo, 1, 3)"),
    ("k_memo9", "substr(memo, 1, 9)", "substr(memo, 1, 9)"),
    ("k_amt2_week",
     "CAST(round(amount / 2, 0) * 2 AS VARCHAR) || '|' || CAST(yearweek(transaction_date) AS VARCHAR)",
     "CAST(round(amount / 2, 0) * 2 AS VARCHAR) || '|' || CAST(yearweek(transaction_date) AS VARCHAR)"),
    ("k_amt2_week4",
     "CAST(round(amount / 2, 0) * 2 AS VARCHAR) || '|' || CAST(yearweek(transaction_date + 4) AS VARCHAR)",
     "CAST(round((amount + 1) / 2, 0) * 2 AS VARCHAR) || '|' || CAST(yearweek(transaction_date) AS VARCHAR)"),
    # The demo's `block_on("unique_id")`, which reaches every true pair because
    # the ids are the ground truth. Kept as a column so cpplink can block on it
    # in the demo track; the matched track leaves it out.
    ("k_uid", "CAST(unique_id AS VARCHAR)", "CAST(unique_id AS VARCHAR)"),
]
KEY_NAMES = [k[0] for k in KEYS]
MATCHED_KEYS = [k for k in KEY_NAMES if k != "k_uid"]

# The demo's SQL blocking rules, verbatim, for the splink demo track.
DEMO_RULES_SQL = [
    """
    strftime(l.transaction_date, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
    """,
    """
    strftime(l.transaction_date+15, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
    """,
    "substr(l.memo,1,9) = substr(r.memo,1,9)",
    """
    round(l.amount/2,0)*2 = round(r.amount/2,0)*2
    and yearweek(r.transaction_date) = yearweek(l.transaction_date)
    """,
    """
    round(l.amount/2,0)*2 = round((r.amount+1)/2,0)*2
    and yearweek(r.transaction_date) = yearweek(l.transaction_date + 4)
    """,
    "l.unique_id = r.unique_id",
]

# The comparison ladders, shared by every configuration. splink's demo declares
# these; cpplink has the same levels (`percentage_within` is splink's
# PercentageDifferenceLevel, strict and over the larger value). The one
# difference is the date: the demo's levels are directed (destination on or
# after origin), cpplink's `date_within` is a window either side, and the
# matched track gives splink the symmetric form so the two agree.
AMOUNT_PERCENTAGES = [0.01, 0.03, 0.1, 0.3]
MEMO_EDITS = [2, 6, 10]
DATE_DAYS = [1, 4, 10, 30]

# cpplink's automatic sources for the native track. A rare value of any key is
# already a candidate through the key's own exact-value source, so what is
# added is a window: neighbours in amount order, on the amount written as a
# zero-padded string so the lexical order is the numeric one, and neighbours in
# memo order, which reaches a memo altered before its ninth character.
NATIVE_EXTRA = [
    {"type": "sorted_neighbourhood", "column": "amount_key", "window": 10},
    {"type": "sorted_neighbourhood", "column": "memo", "window": 10},
]

TRACKS = ("demo", "matched", "native", "directed")
SPLINK_TRACKS = ("demo", "matched", "directed")
CPPLINK_TRACKS = ("demo", "matched", "native")
THRESHOLDS = "0.001,0.01,0.1,0.5,0.9,0.99,0.999"


def cpplink_schema(track):
    levels_amount = [{"type": "null"}, {"type": "exact"}]
    levels_amount += [{"type": "percentage_within", "threshold": p}
                      for p in AMOUNT_PERCENTAGES]
    levels_amount.append({"type": "else"})
    levels_memo = [{"type": "null"}, {"type": "exact"}]
    levels_memo += [{"type": "levenshtein", "threshold": d} for d in MEMO_EDITS]
    levels_memo.append({"type": "else"})
    levels_date = [{"type": "null"}]
    levels_date += [{"type": "date_within", "threshold": d} for d in DATE_DAYS]
    levels_date.append({"type": "else"})

    keys = KEY_NAMES if track == "demo" else MATCHED_KEYS
    blocking = [{"type": "exact_value", "column": k} for k in keys]
    if track == "native":
        blocking += NATIVE_EXTRA
    columns = [
        {"name": "memo", "type": "string"},
        {"name": "transaction_date", "type": "date"},
        {"name": "amount", "type": "double"},
    ]
    columns += [{"name": k, "type": "string"} for k in keys]
    if track == "native":
        # The amount as a zero-padded string, so a lexical window is a numeric one.
        columns.append({"name": "amount_key", "type": "string"})
    return {
        "unique_id": "unique_id",
        "columns": columns,
        "comparisons": [
            {"name": "amount", "columns": ["amount"], "term_frequency": False,
             "levels": levels_amount},
            {"name": "memo", "columns": ["memo"], "term_frequency": False,
             "levels": levels_memo},
            {"name": "transaction_date", "columns": ["transaction_date"],
             "term_frequency": False, "levels": levels_date},
        ],
        "blocking": blocking,
    }


def schema_path(track):
    return os.path.join(SCHEMAS, f"transactions.{track}.json")
