# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The form, built from the schema that the model was estimated under.

A schema states what a search needs a form to know. `comparisons` is an ordered
list of what the model scores, each naming its columns; `columns` carries every
type and every derivation. So the boxes are not a guess and not a fixed list:
they are what the model reads, in the order it reads them.

Three rules do all of it.

A comparison's boxes are **its columns resolved back to their non-derived
source**, deduplicated. The email comparison reads `email` and
`email_username`, the second derived from the first, so it is one box: the core
fills the derived value in from the text that was typed. A location comparison
reading latitude and longitude is two boxes under one heading.

The widget is **the column's type**. A date is a date, a list column is split
into its elements, a number is parsed, and a string is sent as it was typed.

And a string column the file holds few values of is **a dropdown**, whose blank
entry leaves the column out of the query rather than sending a value with. That
is the one rule the schema cannot supply, because a schema does not record that
a gender column holds two values; the data does.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

# A string column with no more distinct values than this is offered as a
# dropdown. Above it, typing is the only sensible way in.
CHOICE_LIMIT = 12

# The blank entry of a dropdown. It is not a value: it leaves the column out of
# the query, which is the null level, and that is what "not stated" means. A
# literal "unknown" would disagree with every record that holds a value.
BLANK = "—"


class FormError(ValueError):
    """A value typed into a box is not what its column holds."""


@dataclass(frozen=True)
class Box:
    """One input: the column it fills, and how it is filled."""

    column: str
    label: str
    kind: str  # "text", "date", "number", "list" or "choice"
    comparison: str
    choices: tuple[str, ...] = ()
    help: str = ""


# What a cell is marked as, from the level its comparison landed on. A level
# that is neither the exact one nor null nor `else` is a partial agreement,
# whatever metric it happens to be written with.
EXACT = "exact"
PARTIAL = "partial"
UNMARKED = ""


@dataclass(frozen=True)
class FormSpec:
    """The boxes to draw, and the columns to show beside a hit."""

    boxes: tuple[Box, ...] = ()
    table_columns: tuple[str, ...] = ()
    id_column: str = ""
    # The columns a dropdown would be offered for, had their values been read.
    list_columns: frozenset[str] = field(default_factory=frozenset)
    # Per comparison: what each of its levels means, and which of the table's
    # columns that comparison speaks for.
    classes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    painted: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def box(self, column: str) -> Box | None:
        for box in self.boxes:
            if box.column == column:
                return box
        return None


def prettify(name: str) -> str:
    """A column name as a label: `last_name` becomes "Last name"."""
    words = re.sub(r"[_\-]+", " ", name).strip()
    return words[:1].upper() + words[1:] if words else name


def transforms_of(spec: Mapping) -> list[str]:
    """The derivation's transforms, however the schema spells them."""
    derive = spec.get("derive") or {}
    chain = derive.get("transforms", derive.get("transform"))
    if chain is None:
        return []
    return list(chain) if isinstance(chain, (list, tuple)) else [str(chain)]


def source_of(columns: Mapping[str, Mapping], name: str) -> str:
    """The real column a derived one comes from, following the chain.

    A derived column is never typed into: it is computed from its source during
    the search, which is why a comparison naming both gets one box.
    """
    seen = set()
    while name in columns and (columns[name].get("derive") or {}).get("from"):
        if name in seen:  # a cycle the schema parser would have refused
            break
        seen.add(name)
        name = columns[name]["derive"]["from"]
    return name


def _kind(column: Mapping) -> str:
    return {
        "date": "date",
        "double": "number",
        "int64": "number",
        "string_list": "list",
    }.get(column.get("type") or "string", "text")


def _class_of(level_type: str) -> str:
    """What a level means for a cell: exact, a partial agreement, or nothing.

    `null` is the absence of a value and `else` is a disagreement, so neither
    is an agreement to mark. Everything between them fired on some predicate
    the schema wrote, which is a partial match whatever metric it uses.
    """
    if level_type == "exact":
        return EXACT
    if level_type in ("null", "else"):
        return UNMARKED
    return PARTIAL


def _help(columns: Mapping[str, Mapping], name: str, kind: str) -> str:
    """What a reader needs to know about this box, from the schema alone."""
    if kind == "list":
        return "Separate the parts with spaces or commas."
    if kind != "text":
        return ""
    # Whether anything derived from this column normalises punctuation away
    # decides whether typing several names with commas can match at all, and
    # the schema is what says so.
    for spec in columns.values():
        derive = spec.get("derive") or {}
        if derive.get("from") == name and "normalize" in transforms_of(spec):
            return "Several values may be separated by commas."
    return ""


def build_form(
    schema: Mapping,
    *,
    choices: Callable[[Sequence[str]], Mapping[str, Sequence[str]]] | None = None,
    labels: Mapping[str, str] | None = None,
    hidden: Iterable[str] = (),
) -> FormSpec:
    """The form the schema describes.

    `choices` is handed the string columns and answers with the values of the
    ones holding few enough to offer as a dropdown; without it every string
    column is typed into.
    """
    columns = {column["name"]: column for column in schema.get("columns", [])}
    labels = dict(labels or {})
    hide = set(hidden)

    ordered: list[tuple[str, str]] = []  # (source column, comparison name)
    for comparison in schema.get("comparisons", []):
        for name in comparison.get("columns", []):
            source = source_of(columns, name)
            if source in hide or source not in columns:
                continue
            if any(source == existing for existing, _ in ordered):
                continue
            ordered.append((source, comparison.get("name", source)))

    strings = [name for name, _ in ordered if _kind(columns[name]) == "text"]
    offered = dict(choices(strings)) if choices and strings else {}

    boxes = []
    for name, comparison in ordered:
        kind = _kind(columns[name])
        values = tuple(str(value) for value in offered.get(name, ()))
        if values:
            kind = "choice"
        boxes.append(
            Box(
                column=name,
                label=labels.get(name) or prettify(name),
                kind=kind,
                comparison=comparison,
                choices=(BLANK, *values) if values else (),
                help=_help(columns, name, kind),
            )
        )

    classes = {}
    painted = {}
    for comparison in schema.get("comparisons", []):
        name = comparison.get("name", "")
        classes[name] = tuple(
            _class_of(level.get("type", "")) for level in comparison.get("levels", [])
        )
        painted[name] = tuple(
            dict.fromkeys(
                source_of(columns, column)
                for column in comparison.get("columns", [])
                if source_of(columns, column) in columns
            )
        )

    id_column = schema.get("unique_id", "unique_id")
    table = [id_column] + [
        name for name in columns if not (columns[name].get("derive") or {})
    ]
    return FormSpec(
        boxes=tuple(boxes),
        table_columns=tuple(dict.fromkeys(table)),
        id_column=id_column,
        classes=classes,
        painted=painted,
        list_columns=frozenset(
            name for name, spec in columns.items() if spec.get("type") == "string_list"
        ),
    )


def _cells(text: str) -> list[str]:
    return [token for token in re.split(r"[,;\s]+", text) if token]


def date_text(value: object) -> str:
    """A date as the `YYYY-MM-DD` text the core is the only reader of."""
    if isinstance(value, datetime.datetime):
        value = value.date()
    if isinstance(value, datetime.date):
        return value.isoformat()
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise FormError(f"{text!r} is not a date: write it as YYYY-MM-DD")
    return text


def number_text(value: object) -> str:
    """A number as text, refused rather than sent where it is not one."""
    text = str(value).strip()
    try:
        float(text)
    except ValueError:
        raise FormError(f"{text!r} is not a number") from None
    return text


def to_query(values: Mapping[str, object], spec: FormSpec) -> dict[str, str | list[str]]:
    """The filled-in form as the record a search takes.

    An empty box is left out rather than sent empty: a column the query does not
    name is *missing*, which is the null level, and that is what an unfilled box
    means.
    """
    query: dict[str, str | list[str]] = {}
    for box in spec.boxes:
        raw = values.get(box.column)
        if raw is None:
            continue
        if box.kind == "date":
            query[box.column] = date_text(raw)
            continue
        text = str(raw).strip()
        if not text or (box.kind == "choice" and text == BLANK):
            continue
        if box.kind == "number":
            query[box.column] = number_text(text)
        elif box.kind == "list":
            cells = _cells(text)
            if cells:
                query[box.column] = cells
        else:
            query[box.column] = text
    return query


def cell_classes(
    spec: FormSpec,
    query: Mapping[str, str | list[str]],
    levels: Mapping[str, int],
) -> dict[str, str]:
    """What each of one hit's cells is, from the levels the pair landed on.

    The level is the only thing that knows: a partial match has no textual
    definition, and a date inside a window or a value agreeing after
    normalisation is an exact level whose two sides do not read alike.

    Only a column the query named is marked. A comparison over a column nobody
    typed into lands on its null level anyway, but one reading two columns --
    a latitude and a longitude -- would otherwise paint the half that was not
    asked about.
    """
    marks: dict[str, str] = {}
    for comparison, level in levels.items():
        ladder = spec.classes.get(comparison, ())
        if level >= len(ladder):
            continue
        mark = ladder[level]
        if not mark:
            continue
        for column in spec.painted.get(comparison, ()):
            if column in query:
                marks[column] = mark
    return marks


def shading(
    frame: pd.DataFrame,
    rows: Sequence[Mapping[str, str]],
    styles: Mapping[str, str],
) -> pd.DataFrame:
    """The frame's own shape, carrying the css each cell's mark asks for.

    Kept beside the values rather than inside them, because a table that says
    what it means by changing its text is a table nothing can read back.
    """
    out = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for position, marks in enumerate(rows):
        if position >= len(out.index):
            break
        for column, mark in marks.items():
            if column in out.columns and mark in styles:
                out.iloc[position, out.columns.get_loc(column)] = styles[mark]
    return out
