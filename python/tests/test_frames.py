# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""A frame in is the file in: one decoder, so the models agree to the byte.

Every input reaches the core through the Arrow C Data Interface, and the
parquet loader goes through the same decoder, so a Linker over
``pd.read_parquet(f)`` must estimate the model the Linker over ``f`` does,
whatever dtypes the frame holds and however it is sliced on the way in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import SEED

import cpplink


def _model_text(linker: cpplink.Linker, path: Path) -> str:
    linker.estimate(out=path, seed=SEED)
    return path.read_text()


def test_a_frame_is_the_file(sample, tmp_path: Path) -> None:
    frame = pd.read_parquet(sample.parquet)
    linker = cpplink.Linker(sample.schema, frame)
    assert linker.records == sample.linker.records
    assert linker.files == []
    assert _model_text(linker, tmp_path / "frame.json") == sample.model_path.read_text()


def test_object_columns_are_sliced_on_the_way_in(sample, tmp_path: Path) -> None:
    # Plain Python strings, a list column and a batch size that forces slices.
    frame = pd.read_parquet(sample.parquet)
    frame["first_name"] = frame["first_name"].astype(object)
    linker = cpplink.Linker(sample.schema, frame, batch_rows=3000)
    assert linker.stats.row_groups > 1, "the frame crossed in more than one batch"
    assert _model_text(linker, tmp_path / "sliced.json") == sample.model_path.read_text()


def test_category_and_arrow_backed_columns(sample, tmp_path: Path) -> None:
    frame = pd.read_parquet(sample.parquet)
    frame["last_name"] = frame["last_name"].astype("category")
    frame["email"] = frame["email"].astype(pd.ArrowDtype(pa.string()))
    linker = cpplink.Linker(sample.schema, frame)
    assert _model_text(linker, tmp_path / "typed.json") == sample.model_path.read_text()


def test_a_pyarrow_table_and_a_reader(sample, tmp_path: Path) -> None:
    table = pq.read_table(sample.parquet)
    assert (
        _model_text(cpplink.Linker(sample.schema, table), tmp_path / "table.json")
        == sample.model_path.read_text()
    )
    reader = pq.ParquetFile(sample.parquet).iter_batches(batch_size=4096)
    stream = pa.RecordBatchReader.from_batches(pq.read_schema(sample.parquet), reader)
    assert (
        _model_text(cpplink.Linker(sample.schema, stream), tmp_path / "reader.json")
        == sample.model_path.read_text()
    )


def test_a_dict_of_frames_is_a_link(link, tmp_path: Path) -> None:
    frames = {"a": pd.read_parquet(link.a), "b": pd.read_parquet(link.b)}
    linker = cpplink.Linker(link.schema, frames)
    assert linker.datasets == 2
    assert linker.mode == link.linker.mode
    assert linker.stats.dataset_rows == link.linker.stats.dataset_rows
    session = linker._session
    last = linker.records - 1
    assert session.dataset_of(0) == "a"
    assert session.dataset_of(last) == "b"
    assert session.qualified_id(last) == "b:" + session.id_of(last)
    assert _model_text(linker, tmp_path / "link.json") == _model_text(
        link.linker, tmp_path / "files.json"
    )


def test_dict_keys_are_checked(sample) -> None:
    frame = pd.read_parquet(sample.parquet)
    with pytest.raises(cpplink.Error, match="comma or a colon"):
        cpplink.Linker(sample.schema, {"a:b": frame, "c": frame})
    with pytest.raises(cpplink.Error, match="is empty"):
        cpplink.Linker(sample.schema, {"": frame, "c": frame})


def test_refusals_name_the_column(sample) -> None:
    frame = pd.read_parquet(sample.parquet)
    with pytest.raises(cpplink.Error, match='column "email" is not in the input'):
        cpplink.Linker(sample.schema, frame.drop(columns=["email"]))
    with pytest.raises(cpplink.Error, match="latitude: expected a floating point"):
        cpplink.Linker(sample.schema, frame.astype({"latitude": str}))
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        cpplink.Linker(sample.schema, 42)
    with pytest.raises(TypeError, match="not both"):
        cpplink.Linker(sample.schema, frame, files=sample.parquet)


def test_a_path_can_be_read_through_pyarrow(sample, tmp_path: Path) -> None:
    # The route a build without Arrow takes for a file: the same rows, the same model.
    inputs = cpplink._inputs(sample.parquet, sample.schema.input_columns, 1 << 20)
    session = cpplink._cpplink._Session.from_streams(sample.schema, inputs)
    assert session.records == sample.linker.records
    linker = cpplink.Linker.__new__(cpplink.Linker)
    linker.schema, linker.files, linker._session = sample.schema, [], session
    assert _model_text(linker, tmp_path / "pyarrow.json") == sample.model_path.read_text()


# ---- what comes out ---------------------------------------------------------


def _sorted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("dataset_a", "id_a", "dataset_b", "id_b") if c in frame.columns]
    return frame.sort_values(keys).reset_index(drop=True)


def test_predict_returns_the_file_it_wrote(sample) -> None:
    frame = sample.frame
    assert list(frame.columns) == [
        "id_a",
        "id_b",
        "gamma",
        "match_weight",
        "match_probability",
    ]
    assert len(frame) == sample.predict_report.predictions
    file = pd.read_parquet(sample.predictions)
    a, b = _sorted_rows(frame), _sorted_rows(file)
    assert a["id_a"].tolist() == b["id_a"].tolist()
    assert a["id_b"].tolist() == b["id_b"].tolist()
    assert a["gamma"].tolist() == b["gamma"].tolist()
    assert a["match_weight"].tolist() == b["match_weight"].tolist()
    # The ids are plain strings in the frame, and a dictionary over every record
    # of the store in the table behind it, which is where zero copy lives.
    assert frame["id_a"].dtype == pd.StringDtype("pyarrow", na_value=float("nan")) or str(
        frame["id_a"].dtype
    ).startswith("str")
    table = pa.table(sample.predict_report.table)
    assert pa.types.is_dictionary(table.schema.field("id_a").type)
    assert len(table["id_a"].chunk(0).dictionary) == sample.linker.records
    assert len(sample.predict_report.table) == len(frame)


def test_predict_without_a_file(sample) -> None:
    frame = sample.linker.predict(sample.model, threshold=10)
    assert sample.linker.last_predict.shards == []
    assert sample.linker.last_predict.merged_path == ""
    assert _sorted_rows(frame).equals(_sorted_rows(sample.frame))
    with pytest.raises(cpplink.Error, match="no out was given"):
        sample.linker.predict(sample.model, threshold=10, format="csv")
    empty = sample.linker.predict(sample.model, threshold=1e9)
    assert len(empty) == 0 and list(empty.columns) == list(frame.columns)


def test_cluster_from_memory_frame_and_file_agree(sample) -> None:
    linker = sample.linker
    linker.predict(sample.model, threshold=10)
    from_memory = linker.cluster()
    memory_report = linker.last_cluster.report
    from_frame = linker.cluster(sample.frame)
    frame_report = linker.last_cluster.report
    from_file = linker.cluster(sample.predictions)
    file_report = linker.last_cluster.report
    assert list(from_memory.columns) == ["unique_id", "cluster_id", "cluster_size"]
    assert memory_report.clusters == frame_report.clusters == file_report.clusters
    assert memory_report.predictions_used == frame_report.predictions_used

    # Which member names a cluster depends on the order the edges arrived, so
    # the partitions are compared as sets of members.
    def partition(frame: pd.DataFrame) -> set[frozenset[str]]:
        return {frozenset(g) for _, g in frame.groupby("cluster_id")["unique_id"]}

    for frame in (from_frame, from_file):
        assert partition(frame) == partition(from_memory)
        assert frame["cluster_size"].sum() == from_memory["cluster_size"].sum()
    # The frame is what the cluster file holds.
    assert len(from_memory) == file_report.written or file_report.written == 0
    assert from_memory["cluster_size"].min() >= 2
    assert set(from_memory["cluster_id"]) <= set(from_memory["unique_id"])


def test_cluster_with_nothing_in_memory_says_so(sample) -> None:
    frame = pd.read_parquet(sample.parquet)
    fresh = cpplink.Linker(sample.schema, frame)
    with pytest.raises(cpplink.Error, match="no predictions in memory"):
        fresh.cluster()
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        fresh.cluster(42)


def test_link_frames_carry_the_datasets(link) -> None:
    linker = cpplink.Linker(
        link.schema, {"a": pd.read_parquet(link.a), "b": pd.read_parquet(link.b)}
    )
    model, _ = linker.estimate(seed=3)
    predictions = linker.predict(model, threshold=10)
    assert list(predictions.columns)[:4] == ["dataset_a", "id_a", "dataset_b", "id_b"]
    assert set(predictions["dataset_a"]) == {"a"}
    assert set(predictions["dataset_b"]) == {"b"}
    clusters = linker.cluster(truth=link.truth)
    assert list(clusters.columns) == [
        "dataset",
        "unique_id",
        "cluster_id",
        "cluster_size",
    ]
    assert set(clusters["dataset"]) == {"a", "b"}
    assert clusters["cluster_id"].str.startswith("a:").all()
    assert linker.last_cluster.quality.f1 > 0.99
    again = linker.cluster(predictions)
    assert len(again) == len(clusters)
    assert linker.last_cluster.report.clusters == clusters["cluster_id"].nunique()


# ---- fixtures without files -------------------------------------------------


def test_gen_sample_frames_are_the_files_rows(sample, tmp_path: Path) -> None:
    from conftest import ROWS

    frame = cpplink.gen_sample(rows=ROWS, truth=tmp_path / "truth.csv")
    file = pd.read_parquet(sample.parquet)
    assert len(frame) == ROWS
    assert list(frame.columns) == list(file.columns)
    assert frame["id"].tolist() == file["id"].tolist()
    assert frame["email"].tolist() == file["email"].tolist()
    assert frame["dob"].tolist() == file["dob"].tolist()
    assert [list(t) for t in frame["address_tokens"]] == [
        list(t) for t in file["address_tokens"]
    ]
    assert (tmp_path / "truth.csv").read_text() == sample.truth.read_text()
    a, b = cpplink.gen_sample(rows=ROWS, out_b=1)
    assert len(a) + len(b) == ROWS
    assert (
        b["id"].tolist()
        == pd.read_parquet(
            cpplink.gen_sample(
                tmp_path / "a.parquet", rows=ROWS, out_b=[tmp_path / "b.parquet"]
            ).replace("a.parquet", "b.parquet")
        )["id"].tolist()
    )


def test_init_from_a_frame_is_init_from_the_file(sample, link) -> None:
    frame = pd.read_parquet(sample.parquet)
    schema, report = cpplink.init(frame)
    assert schema.text == sample.schema.text
    assert [c.role for c in report.columns] == [c.role for c in sample.draft.columns]
    table = pq.read_table(sample.parquet)
    assert cpplink.init(table)[0].text == sample.schema.text
    assert cpplink.init(table.schema)[0].text == sample.schema.text
    both = cpplink.init({"a": pd.read_parquet(link.a), "b": pd.read_parquet(link.b)})
    assert both[0].text == link.schema.text
    with pytest.raises(cpplink.Error, match="is in a but not in b"):
        cpplink.init({"a": frame, "b": frame.drop(columns=["phone"])})
