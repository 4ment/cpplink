# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The finder's resident session: one store, one model, one query at a time.

The two things worth pinning here are the ones that fail quietly. A hit's field
values are read out of the parquet by the row index the hit names, so if the
store ever stopped being in file order the page would draw one person's details
beside another person's score. And a search installs the query as a row in the
store, so two at once are two writers.
"""

from __future__ import annotations

import concurrent.futures
import json

import pytest
from conftest import needs_core_parquet

import cpplink
from cpplink_finder import form, settings
from cpplink_finder.rows import RowFetcher
from cpplink_finder.session import Finder, FinderError


@pytest.fixture(scope="module")
def finder(sample) -> Finder:
    return Finder(sample.schema_path, sample.model_path, sample.parquet)


def _row(sample, row: int) -> dict[str, str]:
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(str(sample.parquet))
    return {
        name: ""
        if table.column(name)[row].as_py() is None
        else str(table.column(name)[row].as_py())
        for name in table.column_names
    }


@needs_core_parquet
def test_the_three_files_load_as_one_session(finder, sample) -> None:
    assert finder.records == sample.linker.records
    # A box per column the schema compares, and the id first in the table.
    # The order is the schema's, which the draft sets by how much a column is
    # worth, so the strongest field is the first box.
    assert [box.column for box in finder.boxes] == [
        "email",
        "phone",
        "dob",
        "last_name",
        "first_name",
        "postcode",
        "address_tokens",
        "gender",
        "latitude",
        "longitude",
    ]
    assert finder.table_columns[0] == finder.id_column
    assert "email_username" not in finder.table_columns


@needs_core_parquet
def test_the_files_own_values_decide_which_boxes_are_dropdowns(finder) -> None:
    """The one thing the schema cannot say, read off the file at load.

    Gender is not a special case here: it is a dropdown because the file holds
    two values of it, and a column holding thousands is typed into.
    """
    gender = finder.form.box("gender")
    assert gender.kind == "choice"
    assert set(gender.choices) - {form.BLANK} <= {"F", "M"}
    assert finder.form.box("last_name").kind == "text"


@needs_core_parquet
def test_a_record_finds_itself_and_brings_its_own_details(finder, sample) -> None:
    row = _row(sample, 0)
    outcome = finder.search(
        {"last_name": row["last_name"], "first_name": row["first_name"]}, k=5
    )
    assert len(outcome) == 5
    top = outcome.hits[0]
    assert top.id == sample.linker.id_of(0)
    assert top.rank == 1
    assert top.values["last_name"] == row["last_name"]
    # Descending weight, and the ledger behind every hit by default.
    weights = [hit.match_weight for hit in outcome.hits]
    assert weights == sorted(weights, reverse=True)
    assert all("Match weight" in hit.waterfall for hit in outcome.hits)


@needs_core_parquet
def test_the_fetched_row_is_the_row_the_hit_names(finder, sample) -> None:
    """The row-order assumption the display rests on, stated as an equality.

    The id is in the file and the id is on the hit, so the two agreeing over
    every hit is exactly "store row i is parquet row i". A break here is
    otherwise invisible: the weights would be right and the names wrong.
    """
    row = _row(sample, 0)
    outcome = finder.search({"last_name": row["last_name"]}, k=20)
    assert len(outcome) > 1
    for hit in outcome.hits:
        assert hit.values[finder.id_column] == hit.id


@needs_core_parquet
def test_the_fetch_finds_a_row_in_any_row_group(sample, tmp_path) -> None:
    """Several row groups, so the bisect is exercised rather than assumed."""
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "grouped.parquet"
    table = pq.read_table(str(sample.parquet))
    pq.write_table(table, str(path), row_group_size=997)
    fetcher = RowFetcher(str(path), [sample.schema.unique_id])
    assert fetcher.num_rows == table.num_rows
    assert fetcher.group_of(0) == 0
    assert fetcher.group_of(1000) == 1
    assert fetcher.group_of(table.num_rows) == -1

    ids = table.column(sample.schema.unique_id).to_pylist()
    wanted = [0, 996, 997, 5000, table.num_rows - 1]
    found = fetcher.fetch(wanted)
    assert [found[row][sample.schema.unique_id] for row in wanted] == [
        str(ids[row]) for row in wanted
    ]
    # A row past the end is left out rather than raising.
    assert fetcher.fetch([table.num_rows + 10]) == {}


@needs_core_parquet
def test_an_empty_form_is_refused_rather_than_scoring_everything(finder) -> None:
    with pytest.raises(FinderError, match="at least one field"):
        finder.search({})
    # And a dropdown left blank is an empty form, because it names no column.
    with pytest.raises(FinderError, match="at least one field"):
        finder.search({"gender": form.BLANK})


@needs_core_parquet
def test_the_prior_is_the_one_a_search_asks(finder, sample) -> None:
    row = _row(sample, 0)
    outcome = finder.search({"last_name": row["last_name"]}, k=3)
    report = outcome.report
    assert report.prior != report.model_prior
    assert report.records == finder.records
    # The model's own prior is still available to compare against.
    # Zero expected matches is how the model's own prior is asked for.
    plain = finder.search({"last_name": row["last_name"]}, k=3, expected_matches=0)
    assert plain.report.prior == plain.report.model_prior
    assert [hit.id for hit in plain.hits] == [hit.id for hit in outcome.hits]


@needs_core_parquet
def test_searches_from_several_threads_do_not_overlap(finder, sample) -> None:
    """The query is a row of the store, so two at once are two writers.

    A page is served to more than one browser session on more than one thread,
    and they share the store. Under no lock this corrupts the store rather than
    raising, so what the test can assert is that every answer is still the right
    one when four of them run at once.
    """
    rows = [_row(sample, index) for index in range(4)]

    def ask(row):
        return finder.search(
            {"last_name": row["last_name"], "first_name": row["first_name"]}, k=3
        )

    # What each query answers on its own, with nothing else running.
    expected = [[hit.id for hit in ask(row).hits] for row in rows]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(ask, rows * 3))
    assert [[hit.id for hit in outcome.hits] for outcome in outcomes] == expected * 3
    assert finder.records == sample.linker.records


@needs_core_parquet
def test_the_settings_are_kept_beside_the_schema(sample, tmp_path) -> None:
    """A settings file holds what the schema has no opinion about, and no more.

    Everything the form draws comes from the schema, so what is saved is the
    run's preferences: how many hits, how many threads, what a score must clear.
    """
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(sample.schema.to_json())
    finder = Finder(schema_path, sample.model_path, sample.parquet)
    assert finder.settings["k"] == 10

    finder.save_settings({"k": 25, "threads": 2, "nonsense": 1})
    kept = json.loads(settings.settings_path(schema_path).read_text())
    assert kept["k"] == 25
    assert kept["threads"] == 2
    assert "nonsense" not in kept

    again = Finder(schema_path, sample.model_path, sample.parquet)
    assert again.settings["k"] == 25
    row = _row(sample, 0)
    assert len(again.search({"last_name": row["last_name"]})) == 25


@needs_core_parquet
def test_a_hidden_column_is_left_off_the_form_but_not_the_table(sample, tmp_path) -> None:
    schema_path = tmp_path / "hidden.json"
    schema_path.write_text(sample.schema.to_json())
    settings.save(schema_path, {**settings.defaults(), "hidden": ["latitude"]})
    finder = Finder(schema_path, sample.model_path, sample.parquet)
    assert finder.form.box("latitude") is None
    assert "latitude" in finder.table_columns


@needs_core_parquet
def test_more_than_one_input_is_refused_with_the_reason(link, tmp_path) -> None:
    schema_path = tmp_path / "link_schema.json"
    schema_path.write_text(link.schema.to_json())
    model_path = tmp_path / "link_model.json"
    model, _ = link.linker.estimate(out=model_path, seed=7)
    assert isinstance(model, cpplink.Model)
    with pytest.raises(FinderError, match="one input"):
        Finder(schema_path, model_path, [link.a, link.b])


@needs_core_parquet
def test_a_hit_carries_the_level_every_comparison_landed_on(finder, sample) -> None:
    """Which is what the results table colours its cells from.

    The levels come off the same waterfall the ledger is printed from, so a
    cell's colour and the bits behind it cannot disagree.
    """
    row = _row(sample, 0)
    outcome = finder.search({"last_name": row["last_name"]}, k=3)
    top = outcome.hits[0]
    assert top.levels
    assert set(top.levels) <= set(sample.schema.comparison_names)
    # The record found by its own surname landed on that comparison's exact
    # level, which is the one below null.
    assert top.levels["last_name"] == 1
    marks = form.cell_classes(finder.form, outcome.query, top.levels)
    assert marks["last_name"] == form.EXACT
    # A comparison the query said nothing about is on its null level, so it
    # marks nothing however the records happen to agree.
    assert "first_name" not in marks

    # Without a ledger there are no levels, and so no colours.
    plain = finder.search({"last_name": row["last_name"]}, k=1, explain=False)
    assert plain.hits[0].levels == {}
