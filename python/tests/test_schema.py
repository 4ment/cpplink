# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cpplink


def test_file_json_dict_round_trip_is_stable(sample) -> None:
    schema = cpplink.Schema.from_file(str(sample.schema_path))
    once = schema.to_json()
    again = cpplink.Schema.from_dict(json.loads(once)).to_json()
    assert once == again
    assert schema.to_dict() == json.loads(once)
    assert schema.unique_id == sample.schema.unique_id
    assert schema.column_names == sample.schema.column_names
    assert schema.comparison_names == sample.schema.comparison_names
    assert len(schema.blocking) == len(sample.schema.blocking)
    assert schema.gamma_width == sample.schema.gamma_width


def test_source_text_is_kept(sample) -> None:
    text = sample.schema_path.read_text()
    schema = cpplink.Schema.from_file(str(sample.schema_path))
    assert schema.text == text


def test_bad_schema_raises_with_the_parser_message() -> None:
    with pytest.raises(cpplink.Error, match="not valid JSON"):
        cpplink.Schema.from_json("{nope")
    with pytest.raises(cpplink.Error) as failure:
        cpplink.Schema.from_dict(
            {
                "columns": [{"name": "a", "type": "string"}],
                "comparisons": [
                    {"name": "a", "columns": ["a"], "levels": [{"type": "exact"}]}
                ],
            }
        )
    assert "else" in str(failure.value)
    assert issubclass(cpplink.Error, RuntimeError)


def test_missing_file_raises() -> None:
    with pytest.raises(cpplink.Error):
        cpplink.Schema.from_file("/nonexistent/schema.json")


def test_save_writes_the_text(sample, tmp_path: Path) -> None:
    out = tmp_path / "copy.json"
    sample.schema.save(str(out))
    assert out.read_text() == sample.schema.text


def test_init_draft_reports_columns(sample) -> None:
    assert sample.draft.path == str(sample.parquet)
    names = [column.name for column in sample.draft.columns]
    assert "email" in names
    roles = {column.name: column.role for column in sample.draft.columns}
    assert roles["email"] == "email"
    assert sample.draft.text.strip() != ""
    assert sample.draft.json == sample.schema.text


def test_init_role_override(sample) -> None:
    schema, draft = cpplink.init(sample.parquet, roles={"last_name": "text"})
    roles = {column.name: (column.role, column.role_given) for column in draft.columns}
    assert roles["last_name"] == ("text", True)
    assert "text" in cpplink.known_roles()
    with pytest.raises(cpplink.Error):
        cpplink.init(sample.parquet, roles={"last_name": "not-a-role"})
