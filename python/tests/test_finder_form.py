# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The form the schema describes, and the query a filled-in one makes.

What is under test is that nothing here is a list of field names: the boxes are
the comparisons' columns, the widgets are the columns' types, and both come out
of the schema file rather than out of this package.
"""

from __future__ import annotations

import datetime
import json
import pathlib

import pytest

from cpplink_finder import form

SAMPLE_SCHEMA = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "sample_schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SAMPLE_SCHEMA.read_text())


@pytest.fixture(scope="module")
def spec(schema) -> form.FormSpec:
    return form.build_form(schema)


def test_a_box_per_compared_column_in_the_schemas_own_order(spec, schema) -> None:
    assert [box.column for box in spec.boxes] == [
        "last_name",
        "first_name",
        "gender",
        "dob",
        "email",
        "phone",
        "postcode",
        "latitude",
        "longitude",
        "address_tokens",
    ]
    # Which is the comparisons' order, and every box says which one it is for.
    assert [box.comparison for box in spec.boxes][:4] == [
        "last_name",
        "first_name",
        "gender",
        "dob",
    ]
    assert spec.box("latitude").comparison == "location"


def test_a_derived_column_is_not_a_box_because_it_is_filled_in_for_you(spec) -> None:
    """The email comparison reads two columns and gets one box.

    `email_username` is derived from `email`, and the core computes it from the
    text that was typed. A box for it would be asking for the same thing twice.
    """
    assert spec.box("email_username") is None
    assert spec.box("email") is not None
    assert spec.box("email").comparison == "email"


def test_the_widget_is_the_columns_type(spec) -> None:
    assert spec.box("last_name").kind == "text"
    assert spec.box("dob").kind == "date"
    assert spec.box("latitude").kind == "number"
    assert spec.box("address_tokens").kind == "list"


def test_a_column_the_file_holds_few_values_of_becomes_a_dropdown(schema) -> None:
    """The one thing the schema cannot say, answered from the data.

    A schema does not record that a gender column holds two values, so the
    values are read off the file and any low-cardinality string column gets the
    same treatment rather than gender being a special case.
    """
    spec = form.build_form(schema, choices=lambda columns: {"gender": ["F", "M"]})
    gender = spec.box("gender")
    assert gender.kind == "choice"
    assert gender.choices == (form.BLANK, "F", "M")
    # And the blank entry leaves the column out rather than sending a value.
    assert form.to_query({"gender": form.BLANK}, spec) == {}
    assert form.to_query({"gender": "F"}, spec) == {"gender": "F"}


def test_the_table_is_the_id_and_every_column_the_file_holds(spec, schema) -> None:
    assert spec.id_column == schema["unique_id"]
    assert spec.table_columns[0] == schema["unique_id"]
    # Derived columns are not in the parquet, so they are not in the table.
    assert "email_username" not in spec.table_columns
    assert "latitude" in spec.table_columns
    assert "address_tokens" in spec.table_columns


def test_a_hidden_column_is_neither_drawn_nor_asked_for(schema) -> None:
    spec = form.build_form(schema, hidden=["latitude", "longitude"])
    assert spec.box("latitude") is None
    assert spec.box("longitude") is None
    # It is still shown beside a hit: not typing it in does not mean not seeing it.
    assert "latitude" in spec.table_columns


def test_a_label_can_be_overridden_and_is_otherwise_the_column_name(schema) -> None:
    assert form.prettify("last_name") == "Last name"
    assert form.prettify("dob") == "Dob"
    spec = form.build_form(schema, labels={"dob": "Date of birth"})
    assert spec.box("dob").label == "Date of birth"
    assert spec.box("last_name").label == "Last name"


def test_an_empty_box_is_left_out_rather_than_sent_empty(spec) -> None:
    query = form.to_query(
        {"last_name": "  smith  ", "first_name": "", "postcode": "   "}, spec
    )
    assert query == {"last_name": "smith"}
    assert form.to_query({}, spec) == {}


def test_a_list_column_is_spread_over_its_elements(spec) -> None:
    query = form.to_query({"address_tokens": "12 Rue Blanche, Paris"}, spec)
    assert query == {"address_tokens": ["12", "Rue", "Blanche", "Paris"]}


def test_a_date_is_the_one_form_the_core_parses(spec) -> None:
    assert form.to_query({"dob": datetime.date(1974, 3, 2)}, spec) == {
        "dob": "1974-03-02"
    }
    assert form.to_query({"dob": "1974-03-02"}, spec) == {"dob": "1974-03-02"}
    with pytest.raises(form.FormError, match="YYYY-MM-DD"):
        form.to_query({"dob": "02/03/1974"}, spec)


def test_a_number_is_refused_rather_than_sent_as_text(spec) -> None:
    assert form.to_query({"latitude": "48.85"}, spec) == {"latitude": "48.85"}
    with pytest.raises(form.FormError, match="not a number"):
        form.to_query({"latitude": "north"}, spec)


def test_the_schema_says_whether_commas_can_be_typed(schema) -> None:
    """Punctuation is undecidable for free text, and the derivations settle it.

    A column something normalises is a column whose commas are turned into
    spaces before anything compares them, so several values in one box can
    match. Without such a derivation they cannot, and the box does not say they
    can.
    """
    plain = form.build_form(schema)
    assert plain.box("first_name").help == ""

    keyed = json.loads(json.dumps(schema))
    keyed["columns"].append(
        {
            "name": "first_name_key",
            "derive": {
                "from": "first_name",
                "transforms": ["normalize", "sorted_tokens"],
            },
        }
    )
    spec = form.build_form(keyed)
    assert "commas" in spec.box("first_name").help
    # And the derived key is still not a box, nor a column of the table.
    assert spec.box("first_name_key") is None
    assert "first_name_key" not in spec.table_columns


def test_a_schema_with_no_comparisons_draws_no_boxes() -> None:
    spec = form.build_form({"unique_id": "id", "columns": [], "comparisons": []})
    assert spec.boxes == ()
    assert spec.table_columns == ("id",)


def test_every_level_is_classed_as_exact_partial_or_nothing(spec) -> None:
    """A partial match is anything that fired and is not the exact level.

    `null` is the absence of a value and `else` is a disagreement, so neither
    is an agreement to mark; everything between them fired on a predicate the
    schema wrote, whatever metric it is written with.
    """
    # last_name: null, exact, jaro_winkler, jaro_winkler, else
    assert spec.classes["last_name"] == (
        form.UNMARKED,
        form.EXACT,
        form.PARTIAL,
        form.PARTIAL,
        form.UNMARKED,
    )
    # address is a list comparison and is classed the same way.
    assert spec.classes["address"][1] == form.EXACT
    assert spec.classes["address"][2] == form.PARTIAL
    # A comparison speaks for the columns it reads, derived ones resolved.
    assert spec.painted["email"] == ("email",)
    assert spec.painted["location"] == ("latitude", "longitude")


def test_a_cell_is_marked_from_the_level_the_pair_landed_on(spec) -> None:
    query = form.to_query({"last_name": "Smith", "postcode": "2000"}, spec)
    exact = form.cell_classes(spec, query, {"last_name": 1, "postcode": 1})
    assert exact == {"last_name": form.EXACT, "postcode": form.EXACT}

    partial = form.cell_classes(spec, query, {"last_name": 2, "postcode": 2})
    # postcode has no level 2 but `else`, so only the name is marked.
    assert partial == {"last_name": form.PARTIAL}

    # Null and else mark nothing at all.
    assert form.cell_classes(spec, query, {"last_name": 0}) == {}
    assert form.cell_classes(spec, query, {"last_name": 4}) == {}


def test_a_column_nobody_typed_into_is_never_marked(spec) -> None:
    """Which is what keeps a two-column comparison honest.

    Location reads a latitude and a longitude. Giving one of them and landing
    on a level would otherwise paint the half that was not asked about.
    """
    query = form.to_query({"latitude": "48.85"}, spec)
    # Location's levels are null, geo_within, geo_within, else: there is no
    # exact one, so the nearest it can be is a partial match.
    assert form.cell_classes(spec, query, {"location": 1}) == {"latitude": form.PARTIAL}
    assert form.cell_classes(spec, {}, {"last_name": 1}) == {}


def test_the_shading_marks_those_cells_and_leaves_the_rest_alone() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [
            {"#": 1, "Probability": 0.99, "last_name": "Smith", "postcode": "2000"},
            {"#": 2, "Probability": 0.40, "last_name": "Smyth", "postcode": "2000"},
        ]
    )
    styles = {form.EXACT: "G", form.PARTIAL: "A"}
    rows = [{"last_name": form.EXACT}, {"last_name": form.PARTIAL}]
    css = form.shading(frame, rows, styles)
    assert css.shape == frame.shape
    assert css["last_name"].tolist() == ["G", "A"]
    # A column that was not marked is untouched, as is every column of totals.
    assert css["postcode"].tolist() == ["", ""]
    assert css["Probability"].tolist() == ["", ""]
