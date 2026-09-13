# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Entry point: launch the page, or draft a schema without one.

    cpplink-studio data.parquet                  # opens the page in a browser
    cpplink-studio data.csv --draft schema.json  # writes the default schema
    cpplink-studio data.csv --draft schema.json --typed data.studio.parquet

The headless form runs the same engine the page does: types proposed from the
file, roles guessed from names and values, the templates' comparisons and
derived columns, and a blocking plan drafted under the candidate budget.
"""

import argparse
import json
import os
import subprocess
import sys

from . import blocking, data, guess, schema, stats


def analyse(ds, pricer=None):
    """Types, stats, probes and roles for every column of a dataset. The value
    counts come from the pricer when one is given, so the page counts once."""
    types = {c: t for c, (t, _) in ds.proposed_types().items()}
    column_stats, probes, roles, reasons = {}, {}, {}, {}
    for column in ds.columns:
        counts, nulls = pricer.counts(column) if pricer else (None, None)
        column_stats[column] = stats.column_stats(ds, column, counts, nulls)
        probes[column] = stats.probe_content(ds, column)
        role, confidence, reason = guess.guess_role(column, probes[column],
                                                    column_stats[column], types[column])
        roles[column] = role
        reasons[column] = (confidence, reason)
        types[column] = guess.type_for_role(role, types[column])
    return types, column_stats, probes, roles, reasons


def guess_unique_id(ds, roles, column_stats):
    for column, role in roles.items():
        if role == "id":
            return column
    for column, st in column_stats.items():
        if st["nulls"] == 0 and st["distinct"] == st["rows"] and \
                ds.duckdb_types[column].upper() not in ("DOUBLE", "FLOAT"):
            return column
    return None


def draft_schema(ds, budget_per_row=500):
    pricer = blocking.Pricer(ds, {})
    types, column_stats, probes, roles, reasons = analyse(ds, pricer)
    unique_id = guess_unique_id(ds, roles, column_stats)
    draft = schema.new_draft(ds, types, roles, unique_id)
    schema.apply_templates(draft)
    included = {c["name"]: c["type"] for c in draft["columns"] if c["include"]}
    included_roles = {c["name"]: c["role"] for c in draft["columns"] if c["include"]}
    pricer.types = included
    plan, plan_reasons = blocking.draft_plan(
        pricer, included_roles, included, schema.to_schema(draft)["columns"], ds.rows,
        budget_per_row, draft["derived"])
    draft["blocking"] = plan
    return draft, reasons, plan_reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", help="csv or parquet file to open")
    ap.add_argument("--draft", metavar="SCHEMA", help="write the default schema and exit")
    ap.add_argument("--typed", metavar="PARQUET",
                    help="with --draft, also write the typed parquet cpplink reads")
    ap.add_argument("--budget", type=float, default=500,
                    help="candidate budget per row for the drafted plan (500)")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args(argv)

    if args.draft:
        if not args.data:
            ap.error("--draft needs a data file")
        ds = data.Dataset(args.data)
        draft, reasons, plan_reasons = draft_schema(ds, args.budget)
        for column in draft["columns"]:
            confidence, reason = reasons[column["name"]]
            print(f"{column['name']:24} {column['type']:12} {column['role']:12} "
                  f"{confidence:7} {reason}")
        for reason in plan_reasons:
            print("blocking:", reason)
        if args.typed:
            types = {c["name"]: c["type"] for c in draft["columns"]}
            include = [c["name"] for c in draft["columns"] if c["include"]]
            id_name = data.write_typed_parquet(ds, args.typed, types, draft["unique_id"],
                                               include=include)
            draft["unique_id"] = id_name
            print(f"wrote {args.typed}")
        with open(args.draft, "w") as handle:
            handle.write(schema.to_json(draft))
        print(f"wrote {args.draft}")
        return 0

    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    command = [sys.executable, "-m", "streamlit", "run", page, "--server.port",
               str(args.port), "--browser.gatherUsageStats", "false", "--"]
    if args.data:
        command.append(args.data)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
