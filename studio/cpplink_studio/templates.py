# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""What a role turns into: the comparison levels, the derived columns and the
blocking sources a column of that role gets by default.

The templates follow ``examples/sample_schema.json`` and the comparison
library splink ships, which is what a user coming from splink expects to
see. They are data, so adding a role is adding an entry; nothing else in the
package knows what a surname is. Thresholds are the schema's own starting
values, and ``cpplink levels`` is the command that places them from the
data afterwards.
"""

import copy

# Levels per role. Every comparison starts with null and ends with else, as the
# parser demands; the order is the model, strongest evidence first.
JW = "jaro_winkler"
LEV = "levenshtein"

COMPARISONS = {
    "first_name": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": LEV, "threshold": 1},
        {"type": JW, "threshold": 0.88}, {"type": "else"}]},
    "surname": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": JW, "threshold": 0.92},
        {"type": JW, "threshold": 0.85}, {"type": "else"}]},
    "full_name": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": JW, "threshold": 0.92},
        {"type": JW, "threshold": 0.80}, {"type": "else"}]},
    "dob": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "date_within", "threshold": 2},
        {"type": "date_within", "threshold": 370, "label": "within a year"},
        {"type": "else"}]},
    "date": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "date_within", "threshold": 7},
        {"type": "else"}]},
    "phone": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": LEV, "threshold": 1},
        {"type": "else"}]},
    "postcode": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": LEV, "threshold": 1},
        {"type": "else"}]},
    "gender": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    "category": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    "city": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": JW, "threshold": 0.90},
        {"type": "else"}]},
    "state": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    "country": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    "address": {"term_frequency": True, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": JW, "threshold": 0.92},
        {"type": JW, "threshold": 0.85}, {"type": "else"}]},
    "number": {"term_frequency": False, "levels": [
        {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    "list": {"term_frequency": False, "levels": [
        {"type": "null"}, {"type": "list_overlap", "threshold": 2},
        {"type": "list_overlap", "threshold": 1}, {"type": "else"}]},
}

# The email comparison ranks the whole address above its username in one gamma
# field, which needs the username as a declared derived column.
EMAIL_LEVELS = [
    {"type": "null"}, {"type": "exact"}, {"type": "exact", "column": "{username}"},
    {"type": JW, "threshold": 0.93},
    {"type": JW, "threshold": 0.93, "column": "{username}"}, {"type": "else"}]

GEO_LEVELS = [
    {"type": "null"}, {"type": "geo_within", "threshold": 1},
    {"type": "geo_within", "threshold": 25}, {"type": "else"}]

# Blocking per role: (source type, knob). Names get rare-value agreement,
# because their commonest values are where the candidates go; identifiers and
# dates get exact agreement, because their groups are small.
BLOCKING = {
    "first_name": ("rare_value", {"max_frequency": 100}),
    "surname": ("rare_value", {"max_frequency": 100}),
    "full_name": ("rare_value", {"max_frequency": 100}),
    "dob": ("exact_value", {}),
    "date": ("exact_value", {}),
    "email": ("exact_value", {}),
    "phone": ("exact_value", {}),
    "postcode": ("rare_value", {"max_frequency": 200}),
    "address": ("rare_value", {"max_frequency": 100}),
}

# Roles a comparison is not worth writing for.
NO_COMPARISON = {"none", "id", "latitude", "longitude"}


def comparison_for(name, role, types, derived_names):
    """The comparison dict for one column of a role, or None. ``types`` is
    {column: cpplink type}; ``derived_names`` maps a column to the derived
    columns declared from it (so the email template can find its username)."""
    if role in NO_COMPARISON:
        return None
    if role == "email":
        username = next((d for d, (src, t) in derived_names.items()
                         if src == name and t == "email_username"), None)
        if username is None:
            levels = [lv for lv in EMAIL_LEVELS if "column" not in lv]
            return {"name": name, "columns": [name], "term_frequency": True,
                    "levels": copy.deepcopy(levels)}
        levels = copy.deepcopy(EMAIL_LEVELS)
        for level in levels:
            if level.get("column") == "{username}":
                level["column"] = username
        return {"name": name, "columns": [name, username], "term_frequency": True,
                "levels": levels}
    template = COMPARISONS.get(role)
    if template is None:
        template = COMPARISONS["category"]
    ctype = types.get(name, "string")
    levels = [lv for lv in copy.deepcopy(template["levels"])
              if level_accepts(lv["type"], ctype)]
    return {"name": name, "columns": [name],
            "term_frequency": template["term_frequency"] and ctype != "double",
            "levels": levels}


def level_accepts(level_type, ctype):
    """Whether a level type can read a column type, mirroring the parser."""
    if level_type in ("null", "else"):
        return True
    if level_type == "exact":
        return ctype != "double"
    if level_type in ("levenshtein", "jaro_winkler"):
        return ctype == "string"
    if level_type == "date_within":
        return ctype == "date"
    if level_type == "numeric_within":
        return ctype == "double"
    if level_type in ("list_overlap", "list_jaccard", "list_levenshtein",
                      "list_jaro_winkler"):
        return ctype == "string_list"
    return False


def geo_comparison(lat, lon):
    return {"name": "location", "columns": [lat, lon], "term_frequency": False,
            "levels": copy.deepcopy(GEO_LEVELS)}


def derived_for(name, role):
    """Derived columns a role wants declared: [(derived name, transform)]."""
    if role == "email":
        return [(name + "_username", "email_username")]
    return []


def blocking_for(name, role, ctype):
    """The default blocking source for a column of a role, or None. A double
    has no discrete agreement, and the plan cannot block on a list."""
    if ctype in ("double", "string_list"):
        return None
    entry = BLOCKING.get(role)
    if entry is None:
        return None
    kind, knobs = entry
    source = {"type": kind, "column": name}
    source.update(knobs)
    return source
