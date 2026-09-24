# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The page itself, driven the way a person drives it.

Streamlit's own test runner executes the script and reports what it drew, so
what is checked here is that the page loads the three files, that filling a box
and pressing the button produces the hits, and that an empty form is turned
away rather than scoring every record.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from conftest import needs_core_parquet

APP = pathlib.Path(__file__).resolve().parents[1] / "cpplink_finder" / "app.py"


@pytest.fixture
def page(sample, monkeypatch):
    """The page, run once, with the three files named on the command line."""
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            str(sample.parquet),
            "--schema",
            str(sample.schema_path),
            "--model",
            str(sample.model_path),
        ],
    )
    app = testing.AppTest.from_file(str(APP), default_timeout=120)
    return app.run()


def _value(sample, row: int, column: str) -> str:
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(str(sample.parquet), columns=[column])
    return str(table.column(column)[row].as_py())


@needs_core_parquet
def test_the_page_loads_the_three_files(page, sample) -> None:
    assert not page.exception
    assert any(
        f"{sample.linker.records:,} records" in caption.value for caption in page.caption
    )
    # A box per column the schema compares, labelled by the column's name.
    labels = [box.label for box in page.text_input]
    assert "Last name" in labels
    assert "First name" in labels
    assert "Email" in labels
    # A derived column is filled in during the search, so it is not a box.
    assert "Email username" not in labels
    # The date column is a date picker and the two-valued one a dropdown.
    assert [box.label for box in page.date_input] == ["Dob"]
    assert "Gender" in [box.label for box in page.selectbox]


@needs_core_parquet
def test_filling_a_box_and_pressing_search_draws_the_hits(page, sample) -> None:
    surname = _value(sample, 0, "last_name")
    for box in page.text_input:
        if box.label == "Last name":
            box.set_value(surname)
    after = page.button[0].click().run()
    assert not after.exception
    assert len(after.dataframe) == 1

    # Streamlit's test runner unwraps the Styler, so what it hands back is the
    # data; `shading` is what carries the marking and is tested on its own.
    drawn = after.dataframe[0].value
    frame = getattr(drawn, "data", drawn)
    assert list(frame["#"]) == list(range(1, len(frame) + 1))
    assert frame["Match weight"].is_monotonic_decreasing
    assert frame["last_name"].iloc[0] == surname
    assert frame[sample.schema.unique_id].iloc[0] == sample.linker.id_of(0)
    # The table is the id and every column the file holds, derived ones aside.
    assert list(frame.columns)[:3] == ["#", "Match weight", "Probability"]
    assert list(frame.columns)[3] == sample.schema.unique_id
    assert "email_username" not in frame.columns
    assert "latitude" in frame.columns


@needs_core_parquet
def test_an_empty_form_is_turned_away_rather_than_run(page) -> None:
    after = page.button[0].click().run()
    assert not after.exception
    assert not after.dataframe
    assert any("at least one field" in warning.value for warning in after.warning)


@needs_core_parquet
def test_a_page_with_no_files_asks_for_them(monkeypatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(sys, "argv", ["app.py"])
    app = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("Give a schema" in info.value for info in app.info)
