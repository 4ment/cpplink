# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The page renders every tab over the sample without raising, and the buttons
that reach the binary produce their reports."""

import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "cpplink_studio", "app.py")


@pytest.fixture
def app(sample):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.sidebar.text_input[0].set_value(sample["parquet"])
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    return at


def test_loads_and_renders(app):
    labels = [m.value for m in app.metric]
    assert any("20,000" in v for v in labels)
    # Every tab rendered: the schema tab's JSON names the derived column.
    assert any("email_username" in c.value for c in app.code)


def test_prices_with_binary(app):
    buttons = {b.label: b for b in app.button}
    buttons["Price with cpplink (exact, seconds)"].click().run()
    assert not app.exception, app.exception
    assert any("explain-blocking" in c.value for c in app.caption)


def test_validates_schema(app):
    buttons = {b.label: b for b in app.button}
    buttons["Validate with cpplink inspect"].click().run()
    assert not app.exception, app.exception
    assert any("Records" in c.value or "rows" in c.value.lower() for c in app.code)


def test_csv_goes_through_the_typed_parquet(sample, dataset, tmp_path):
    """A csv cannot reach the binary until the Data tab writes the typed
    parquet; after that, pricing with the binary works."""
    from streamlit.testing.v1 import AppTest
    csv_path = str(tmp_path / "sample.csv")
    dataset.query(f"COPY (SELECT * EXCLUDE (address_tokens), "
                  f"array_to_string(address_tokens, '|') AS address_tokens FROM data) "
                  f"TO '{csv_path}' (FORMAT CSV, HEADER)")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.text_input[0].set_value(csv_path)
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert any("write the typed parquet" in i.value for i in at.info)
    buttons = {b.label: b for b in at.button}
    buttons["Write typed parquet"].click().run()
    assert not at.exception, at.exception
    assert any("the binary now reads this file" in s.value for s in at.success)
    buttons = {b.label: b for b in at.button}
    buttons["Price with cpplink (exact, seconds)"].click().run()
    assert not at.exception, at.exception
    assert any("explain-blocking" in c.value for c in at.caption)
