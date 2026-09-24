# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""One store, one model, one query at a time.

The page is a front end onto a resident `Linker`: the store is loaded once and
every search runs against it, which is the whole reason this is a served page
rather than a command line. Three things follow, and they are why this module
exists rather than the page doing it inline.

A search is **serialised**. The query is installed as a row in the store for as
long as it takes to answer it, so two searches at once are two writers to one
store. Streamlit runs each browser session on its own thread and a cached
resource is shared between them, so that is a real race and not a theoretical
one. The lock is held across the search *and* the ledgers, because the query
stops being a row when the call returns.

The prior is the **search** prior. A model's own prior is the match rate over
the pair space, which is the question deduplication asks; a person asking who in
this file is this person is asking a different one, and the default here says so.

And the hits' field values come from the parquet rather than from a frame held
beside the store.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cpplink

from . import form, settings
from .rows import RowFetcher


class FinderError(RuntimeError):
    """Something about the three files does not hold together."""


# Searching reads one input, because the query is installed as a row of the
# store and a second input would make it a dataset of its own. The refusal is
# before the load rather than after it: there is no point reading a file the
# size this page is built for only to turn it down.
ONE_INPUT = (
    "the finder reads one input: a query row would have to be a dataset of its own"
)


def _one_input(data: object) -> str:
    """The one parquet path, from a path or a sequence holding exactly one."""
    if isinstance(data, (str, os.PathLike)):
        return str(data)
    if isinstance(data, Sequence):
        if len(data) != 1:
            raise FinderError(ONE_INPUT)
        return str(data[0])
    raise FinderError(f"{data!r} is not a path to a parquet file")


@dataclass
class Hit:
    """One record the query scored against, as the page draws it."""

    rank: int
    row: int
    id: str
    match_weight: float
    match_probability: float
    values: dict[str, str] = field(default_factory=dict)
    waterfall: str = ""
    # The level each comparison landed on, which is what decides a cell's
    # colour. Empty where the search was run without explaining itself.
    levels: dict[str, int] = field(default_factory=dict)


@dataclass
class Outcome:
    """What one search produced, and what it cost."""

    hits: list[Hit]
    query: dict[str, str | list[str]]
    report: object
    seconds: float

    def __len__(self) -> int:
        return len(self.hits)


class Finder:
    """A schema, a model and a parquet, loaded and ready to be asked about."""

    def __init__(
        self,
        schema_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
        data_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    ) -> None:
        self.schema_path = str(schema_path)
        self.model_path = str(model_path)
        self.data_path = _one_input(data_path)
        self._lock = threading.Lock()

        self.schema = cpplink.Schema.from_file(self.schema_path)
        self.model = cpplink.Model.from_file(self.model_path)
        self.linker = cpplink.Linker(self.schema, self.data_path)
        if self.linker.datasets > 1:
            raise FinderError(ONE_INPUT)

        self.settings = settings.load(self.schema_path)
        spec = self.schema.to_dict()
        # Only what the file holds can be shown: a derived column is computed
        # during the search and is not in the parquet.
        self._rows = RowFetcher(self.data_path, list(self.schema.input_columns))
        self.form = form.build_form(
            spec,
            choices=self._choices,
            labels=self.settings.get("labels"),
            hidden=self.settings.get("hidden", ()),
        )

    def _choices(self, columns: Sequence[str]) -> dict[str, list[str]]:
        """Which string columns the file holds few enough values of to offer."""
        return self._rows.categorical(columns, limit=form.CHOICE_LIMIT)

    # -- what the page needs to draw itself -------------------------------

    @property
    def records(self) -> int:
        return self.linker.records

    @property
    def id_column(self) -> str:
        return self.form.id_column

    @property
    def boxes(self) -> tuple[form.Box, ...]:
        return self.form.boxes

    @property
    def table_columns(self) -> tuple[str, ...]:
        return self.form.table_columns

    def save_settings(self, values: Mapping) -> None:
        """Keep the run's preferences, which is all a settings file holds."""
        self.settings.update({k: v for k, v in values.items() if k in self.settings})
        settings.save(self.schema_path, self.settings)

    def build_query(self, values: Mapping[str, object]) -> dict[str, str | list[str]]:
        return form.to_query(values, self.form)

    # -- the search --------------------------------------------------------

    def search(
        self,
        values: Mapping[str, object],
        *,
        k: int | None = None,
        threads: int | None = None,
        expected_matches: float | None = None,
        min_probability: float | None = None,
        explain: bool = True,
    ) -> Outcome:
        """The k records scoring highest against a filled-in form.

        Anything not given comes from the settings. `expected_matches` is the
        prior stated as the number of records here expected to be the person
        asked about; it shifts every hit by the same constant, so it moves the
        probability and never the order.
        """
        query = self.build_query(values)
        if not query:
            raise FinderError("fill in at least one field")

        if expected_matches is None:
            expected_matches = self.settings["expected_matches"]
        if min_probability is None:
            min_probability = self.settings["min_probability"]
        # Zero is how "leave them alone" is said for both: no floor on the
        # score, and the model's own prior rather than a stated one.
        expected_matches = expected_matches or None
        min_probability = min_probability or None

        started = time.perf_counter()
        with self._lock:
            result = self.linker.search(
                query,
                self.model,
                k=k or self.settings["k"],
                threads=threads or self.settings["threads"],
                expected_matches=expected_matches,
                probability=min_probability,
                explain=explain,
            )
            ledgers = [w.text for w in result.waterfalls] if explain else []
            landed = (
                [{step.name: step.level for step in w.steps} for w in result.waterfalls]
                if explain
                else []
            )
        seconds = time.perf_counter() - started

        hits = [
            Hit(
                rank=index + 1,
                row=hit.row,
                id=hit.id,
                match_weight=hit.match_weight,
                match_probability=hit.match_probability,
                waterfall=ledgers[index] if index < len(ledgers) else "",
                levels=landed[index] if index < len(landed) else {},
            )
            for index, hit in enumerate(result.hits)
        ]
        self._fill_values(hits)
        return Outcome(hits=hits, query=query, report=result.report, seconds=seconds)

    def _fill_values(self, hits: Sequence[Hit]) -> None:
        """The records behind the hits, off the file, in one pass per group."""
        if not hits:
            return
        found = self._rows.fetch(hit.row for hit in hits)
        for hit in hits:
            hit.values = found.get(hit.row, {})
