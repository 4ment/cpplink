# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""A draft round-trips through the schema JSON, and the JSON the templates
write is one the binary's parser accepts."""

import json

from cpplink_studio import binary, data, guess, schema, stats


def test_draft_from_roles_is_accepted_by_binary(dataset, sample):
    types = {c: t for c, (t, _) in dataset.proposed_types().items()}
    roles = {}
    for column in dataset.columns:
        st = stats.column_stats(dataset, column)
        probe = stats.probe_content(dataset, column)
        roles[column] = guess.guess_role(column, probe, st, types[column])[0]
    draft = schema.new_draft(dataset, types, roles, unique_id="id")
    schema.apply_templates(draft)
    assert schema.problems(draft) == []
    text = schema.to_json(draft)
    parsed = json.loads(text)
    names = {c["name"] for c in parsed["columns"]}
    assert "email_username" in names and "id" not in names
    assert any(c["columns"] == ["latitude", "longitude"] for c in parsed["comparisons"])
    out, err, _ = binary.inspect(text, [sample["parquet"]])
    assert "email_username" in out


def test_from_schema_round_trip(dataset, sample_schema):
    draft = schema.from_schema(sample_schema, dataset)
    again = schema.to_schema(draft)
    assert again["comparisons"] == sample_schema["comparisons"]
    assert again["blocking"] == sample_schema["blocking"]
    assert sorted(c["name"] for c in again["columns"]) == \
        sorted(c["name"] for c in sample_schema["columns"])
    assert schema.problems(draft) == []


def test_problems_catch_undeclared_and_unordered():
    draft = {"unique_id": "id", "columns": [{"name": "a", "type": "string",
                                            "role": "none", "include": True}],
             "derived": [], "blocking": [{"type": "exact_value", "column": "b"}],
             "comparisons": [{"name": "a", "columns": ["a"], "levels": [
                 {"type": "exact"}, {"type": "null"}]}], "cast_options": {}}
    found = schema.problems(draft)
    assert any("names 'b'" in p for p in found)
    assert any("start with a null" in p for p in found)
    assert any("end with an else" in p for p in found)


def test_csv_round_trip_writes_what_the_binary_reads(dataset, sample, tmp_path):
    csv_path = str(tmp_path / "sample.csv")
    dataset.query(f"COPY (SELECT * EXCLUDE (address_tokens), "
                  f"array_to_string(address_tokens, '|') AS address_tokens FROM data) "
                  f"TO '{csv_path}' (FORMAT CSV, HEADER)")
    ds = data.Dataset(csv_path)
    assert ds.rows == dataset.rows
    types = {c: t for c, (t, _) in ds.proposed_types().items()}
    assert types["dob"] == "date"
    # The sniffed types are DuckDB's; an id read as an integer must still be
    # written as a string, and the list column split on its separator.
    types["address_tokens"] = "string_list"
    out = str(tmp_path / "typed.parquet")
    options = {"address_tokens": {"list_separator": "|"}}
    id_name = data.write_typed_parquet(ds, out, types, "id", options)
    assert id_name == "id"
    written = data.Dataset(out)
    assert not written.needs_cast("id", "string")
    assert not written.needs_cast("address_tokens", "string_list")
    assert not written.needs_cast("dob", "date")
    schema_text = json.dumps({"unique_id": "id", "columns": [
        {"name": "dob", "type": "date"}, {"name": "address_tokens", "type": "string_list"},
        {"name": "gender", "type": "string"}]})
    text, _, _ = binary.inspect(schema_text, [out])
    assert "address_tokens" in text
