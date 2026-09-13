# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The draft schema: one dict the page edits and the file cpplink reads.

A draft holds the file's columns with the type and role picked for each, the
derived columns, the comparisons, the blocking plan and the id column. The
schema JSON is built from it by ``to_schema``; ``from_schema`` reads one back
so an existing schema can be opened and edited. The binary's parser is the
validator (``cpplink inspect`` rejects a bad schema before opening the file),
so the only checks here are the ones that keep the editor coherent: a
comparison must name declared columns, and every one starts at null and ends
at else.
"""

import copy
import json

from .templates import blocking_for, comparison_for, derived_for, geo_comparison


def new_draft(ds, types, roles, unique_id=None):
    """A draft from the file: every column included with its proposed type,
    the templates' comparisons, derived columns and blocking for its role."""
    draft = {
        "source": ds.path,
        "unique_id": unique_id,
        "columns": [],
        "derived": [],
        "comparisons": [],
        "blocking": [],
        "cast_options": {},
    }
    for column in ds.columns:
        draft["columns"].append({
            "name": column, "type": types[column], "role": roles.get(column, "none"),
            "include": column != unique_id and roles.get(column) != "id",
        })
    return draft


def apply_templates(draft):
    """Replace the derived columns, comparisons and blocking with what the
    roles imply. Called when the user asks for the defaults again, never
    silently over their edits."""
    types = {c["name"]: c["type"] for c in draft["columns"] if c["include"]}
    roles = {c["name"]: c["role"] for c in draft["columns"] if c["include"]}
    draft["derived"] = []
    for name, role in roles.items():
        for derived_name, transform in derived_for(name, role):
            draft["derived"].append({"name": derived_name, "from": name,
                                     "transform": transform})
    derived_names = {d["name"]: (d["from"], d["transform"]) for d in draft["derived"]}
    comparisons = []
    lat = next((n for n, r in roles.items() if r == "latitude"), None)
    lon = next((n for n, r in roles.items() if r == "longitude"), None)
    for name, role in roles.items():
        if role in ("latitude", "longitude"):
            continue
        comparison = comparison_for(name, role, types, derived_names)
        if comparison is not None:
            comparisons.append(comparison)
    if lat and lon and types.get(lat) == "double" and types.get(lon) == "double":
        comparisons.append(geo_comparison(lat, lon))
    draft["comparisons"] = comparisons
    blocking = []
    for name, role in roles.items():
        source = blocking_for(name, role, types[name])
        if source is not None:
            blocking.append(source)
    for d in draft["derived"]:
        if d["transform"] == "email_username":
            blocking.append({"type": "exact_value", "column": d["name"]})
    draft["blocking"] = blocking
    return draft


def to_schema(draft):
    """The schema JSON as a dict."""
    schema = {}
    if draft.get("unique_id"):
        schema["unique_id"] = draft["unique_id"]
    columns = []
    for c in draft["columns"]:
        if c["include"]:
            columns.append({"name": c["name"], "type": c["type"]})
    for d in draft["derived"]:
        entry = {"name": d["name"], "derive": {"from": d["from"],
                                              "transform": d["transform"]}}
        columns.append(entry)
    schema["columns"] = columns
    schema["comparisons"] = copy.deepcopy(draft["comparisons"])
    schema["blocking"] = copy.deepcopy(draft["blocking"])
    return schema


def to_json(draft):
    return json.dumps(to_schema(draft), indent=2) + "\n"


def from_schema(schema, ds=None):
    """A draft from an existing schema, for editing. Columns the file does not
    hold are kept and flagged; roles are unknown and read as `none`."""
    file_columns = list(ds.columns) if ds is not None else []
    draft = {
        "source": ds.path if ds is not None else None,
        "unique_id": schema.get("unique_id"),
        "columns": [],
        "derived": [],
        "comparisons": copy.deepcopy(schema.get("comparisons", [])),
        "blocking": copy.deepcopy(schema.get("blocking", [])),
        "cast_options": {},
    }
    declared = {}
    for c in schema.get("columns", []):
        if "derive" in c:
            transform = c["derive"].get("transform")
            if isinstance(transform, list) and len(transform) == 1:
                transform = transform[0]
            draft["derived"].append({"name": c["name"], "from": c["derive"]["from"],
                                     "transform": transform})
        else:
            declared[c["name"]] = c.get("type", "string")
    for name in file_columns:
        draft["columns"].append({
            "name": name, "type": declared.get(name, "string"), "role": "none",
            "include": name in declared, "missing_in_file": False})
    for name, ctype in declared.items():
        if name not in file_columns:
            draft["columns"].append({"name": name, "type": ctype, "role": "none",
                                     "include": True, "missing_in_file": True})
    return draft


def problems(draft):
    """Editor-level checks, as a list of messages. Empty means the JSON is
    worth handing to the binary."""
    out = []
    names = {c["name"] for c in draft["columns"] if c["include"]}
    names |= {d["name"] for d in draft["derived"]}
    types = {c["name"]: c["type"] for c in draft["columns"] if c["include"]}
    if not names:
        out.append("no column is included")
    for d in draft["derived"]:
        if d["from"] not in types:
            out.append(f"derived column {d['name']!r} is from {d['from']!r}, "
                       f"which is not included")
    for comparison in draft["comparisons"]:
        for column in comparison.get("columns", []):
            if column not in names:
                out.append(f"comparison {comparison.get('name')!r} names {column!r}, "
                           f"which is not declared")
        levels = comparison.get("levels", [])
        if not levels or levels[0].get("type") != "null":
            out.append(f"comparison {comparison.get('name')!r} must start with a null level")
        if not levels or levels[-1].get("type") != "else":
            out.append(f"comparison {comparison.get('name')!r} must end with an else level")
    for source in draft["blocking"]:
        if source["type"] != "all_pairs" and source.get("column") not in names:
            out.append(f"blocking source {source['type']} names {source.get('column')!r}, "
                       f"which is not declared")
        if source["type"] != "all_pairs" and types.get(source.get("column")) == "double":
            out.append(f"blocking on {source.get('column')!r}, a double column, "
                       f"needs discrete agreement")
        if source["type"] != "all_pairs" and types.get(source.get("column")) == "string_list":
            out.append(f"blocking on {source.get('column')!r}, a list column, "
                       f"is not supported")
        if source["type"] == "minhash" and types.get(source.get("column")) != "string" \
                and source.get("column") in types:
            out.append(f"minhash on {source.get('column')!r} needs a string column")
    if draft.get("unique_id") and draft["unique_id"] in names:
        out.append("the unique_id column should not also be a compared column")
    return out
