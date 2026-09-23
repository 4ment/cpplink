# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""`Linker.search`: top-k retrieval over the weight, from the resident session.

The query is a record of the store only while the search runs, so what this
checks above all is that nothing is left behind afterwards -- neither a row, nor
a moved buffer that a zero-copy table was pointing at.
"""

from __future__ import annotations

import json
import math

import pytest
from conftest import needs_core_parquet

import cpplink


def _value(sample, row: int, column: str) -> str:
    """One cell of the sample file, as the text a caller would type."""
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(str(sample.parquet), columns=[column])
    return str(table.column(column)[row].as_py())


@needs_core_parquet
def test_a_record_finds_itself(sample) -> None:
    query = {
        "last_name": _value(sample, 0, "last_name"),
        "dob": _value(sample, 0, "dob"),
        "postcode": _value(sample, 0, "postcode"),
    }
    hits = sample.linker.search(query, sample.model, k=5)
    assert len(hits) == 5
    assert hits[0].id == sample.linker.id_of(0)
    assert hits[0].match_weight > 0
    # Descending weight, and the probability is the weight through the logistic.
    weights = [hit.match_weight for hit in hits]
    assert weights == sorted(weights, reverse=True)
    assert hits[0].match_probability == pytest.approx(
        cpplink.probability_for_weight(hits[0].match_weight)
    )


@needs_core_parquet
def test_the_report_counts_the_three_kinds_of_comparison(sample) -> None:
    result = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")}, sample.model, k=3
    )
    report = result.report
    assert report.records == sample.linker.records
    # Every comparison is one of the three, and the ones the query says nothing
    # about are free.
    total = report.tabulated + report.evaluated + report.constant
    assert total == len(sample.schema.to_dict()["comparisons"])
    assert report.constant > 0
    assert report.values_walked > 0
    assert report.rescored <= report.records
    assert report.threads == 1
    assert report.prior == report.model_prior


@needs_core_parquet
def test_explain_gives_the_ledger_behind_every_hit(sample) -> None:
    result = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")},
        sample.model,
        k=4,
        explain=True,
    )
    assert len(result.waterfalls) == len(result.hits)
    for hit, waterfall in zip(result.hits, result.waterfalls, strict=True):
        assert waterfall.gamma == hit.gamma
        assert waterfall.weight == pytest.approx(hit.match_weight)
        # The store's record is the first side, the query the second.
        assert waterfall.row_a == hit.row
        assert waterfall.id_a == hit.id
        assert "Match weight" in waterfall.text
    # Without it there is nothing to explain, which is the default.
    plain = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")}, sample.model, k=4
    )
    assert plain.waterfalls == []


@needs_core_parquet
def test_the_prior_shifts_every_hit_and_reorders_nothing(sample) -> None:
    query = {"last_name": _value(sample, 0, "last_name")}
    plain = sample.linker.search(query, sample.model, k=6)
    shifted = sample.linker.search(query, sample.model, k=6, expected_matches=1)
    assert shifted.report.prior != plain.report.prior
    move = shifted.report.prior - plain.report.prior
    assert [hit.id for hit in shifted] == [hit.id for hit in plain]
    for before, after in zip(plain, shifted, strict=True):
        assert after.match_weight == pytest.approx(before.match_weight + move)
    # And it is the odds of one record in the store being the one asked about.
    n = sample.linker.records
    assert shifted.report.prior == pytest.approx(math.log2(1.0 / (n - 1.0)))


@needs_core_parquet
def test_a_threshold_can_refuse_every_record(sample) -> None:
    result = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")},
        sample.model,
        k=10,
        threshold=1e6,
    )
    assert len(result) == 0
    assert "no record here is this one" in result.text


@needs_core_parquet
def test_threads_change_nothing(sample) -> None:
    query = {
        "last_name": _value(sample, 0, "last_name"),
        "dob": _value(sample, 0, "dob"),
    }
    one = sample.linker.search(query, sample.model, k=8, threads=1)
    many = sample.linker.search(query, sample.model, k=8, threads=4)
    assert [(h.row, h.match_weight) for h in one] == [
        (h.row, h.match_weight) for h in many
    ]
    assert many.report.threads == 4


@needs_core_parquet
def test_a_column_the_query_omits_is_missing_not_empty(sample) -> None:
    # None is the same as leaving the column out, and both leave that
    # comparison on its null level for every record, which is free.
    named = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name"), "postcode": None},
        sample.model,
        k=3,
    )
    omitted = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")}, sample.model, k=3
    )
    assert [h.id for h in named] == [h.id for h in omitted]
    assert named.report.constant == omitted.report.constant


@needs_core_parquet
def test_the_store_is_left_as_it_was_found(sample) -> None:
    """The query is a record for the length of the search and no longer.

    The zero-copy tables the bindings hand out are borrowed pointers into the
    store's id arena, so a query row that moved one would leave them dangling.
    The store reserves room for it at load, and this is the test of that: a
    table exported before the search still reads after it.
    """
    pa = pytest.importorskip("pyarrow")
    sample.linker.predict(sample.model, threshold=10)
    table = pa.table(sample.linker.last_predict.table)
    before = table.column("id_a").to_pylist()
    records = sample.linker.records

    sample.linker.search(
        {"last_name": "a-surname-this-store-has-never-held"}, sample.model, k=5
    )

    assert sample.linker.records == records
    assert table.column("id_a").to_pylist() == before


@needs_core_parquet
def test_a_value_the_store_never_held_is_still_compared(sample) -> None:
    result = sample.linker.search(
        {"last_name": "zzzzzzzzznotasurname"}, sample.model, k=3
    )
    assert result.report.values_adopted >= 1
    assert len(result) == 3
    # And the next query is unaffected by it.
    again = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")}, sample.model, k=3
    )
    assert again.report.values_adopted == 0
    assert again[0].id == sample.linker.id_of(0)


@needs_core_parquet
def test_a_list_or_tuple_is_spread_over_the_repeats(sample) -> None:
    # address_tokens is the sample's list column; two elements are two fields.
    result = sample.linker.search({"address_tokens": ["one", "two"]}, sample.model, k=2)
    assert len(result) == 2


@needs_core_parquet
def test_json_carries_the_hits(sample) -> None:
    result = sample.linker.search(
        {"last_name": _value(sample, 0, "last_name")}, sample.model, k=3
    )
    parsed = json.loads(result.json())
    assert parsed["records"] == sample.linker.records
    assert len(parsed["hits"]) == 3
    assert parsed["hits"][0]["id"] == result[0].id
    assert parsed["hits"][0]["match_weight"] == pytest.approx(result[0].match_weight)


@needs_core_parquet
def test_an_empty_query_and_a_bad_date_are_refused(sample) -> None:
    with pytest.raises(cpplink.Error, match="at least one column"):
        sample.linker.search({}, sample.model)
    with pytest.raises(cpplink.Error, match="YYYY-MM-DD"):
        sample.linker.search({"dob": "24/01/1979"}, sample.model)
    with pytest.raises(cpplink.Error, match="not both"):
        sample.linker.search(
            {"last_name": "x"}, sample.model, expected_matches=1, prior_weight=0.0
        )


@needs_core_parquet
def test_link_mode_is_refused_with_the_reason(link) -> None:
    model, _ = link.linker.estimate()
    with pytest.raises(cpplink.Error, match="one input"):
        link.linker.search({"last_name": "smith"}, model)


@needs_core_parquet
def test_matches_the_command_line(sample) -> None:
    """The binding and `cpplink search` are one implementation, so the hits and
    the printed report are the same."""
    query = {
        "last_name": _value(sample, 0, "last_name"),
        "dob": _value(sample, 0, "dob"),
        "postcode": _value(sample, 0, "postcode"),
    }
    result = sample.linker.search(query, sample.model, k=5)
    args = [
        "search",
        "--schema",
        str(sample.schema_path),
        "--model",
        str(sample.model_path),
        str(sample.parquet),
        "-k",
        "5",
    ]
    for column, value in query.items():
        args += ["--field", f"{column}={value}"]
    cli = cpplink.run(args)
    assert cli.code == 0, cli.stderr
    for hit in result:
        assert hit.id in cli.stdout
    # The two reports differ only in what the timings read.
    body = result.text.split("Record ", 1)[1]
    assert body == cli.stdout.split("Record ", 1)[1]
