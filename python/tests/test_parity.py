# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The load-bearing test: the Linker and the command line agree to the byte.

Both call the same core functions through the same pipeline glue, so a model
estimated from Python with a seed is the model `cpplink estimate --seed` writes,
the predictions are the same rows, and the clusters are the same partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

import cpplink

from conftest import SEED


def _run(*args: str) -> cpplink.CliResult:
    result = cpplink.run([str(arg) for arg in args])
    assert result.code == 0, result.stderr
    return result


def _sorted(path: Path):
    return pq.read_table(str(path)).sort_by(
        [("id_a", "ascending"), ("id_b", "ascending")]
    )


def test_estimate_parity(sample, tmp_path: Path) -> None:
    cli_model = tmp_path / "model.json"
    result = _run(
        "estimate",
        "--schema",
        sample.schema_path,
        "--seed",
        SEED,
        "--out",
        cli_model,
        sample.parquet,
    )
    assert cli_model.read_text() == sample.model_path.read_text()
    # The printed parameter table is the same too; the estimate report carries
    # timings and is not compared.
    assert sample.model.text in result.stdout


def test_predict_parity(sample, tmp_path: Path) -> None:
    cli_out = tmp_path / "predictions.parquet"
    result = _run(
        "predict",
        "--schema",
        sample.schema_path,
        "--model",
        sample.model_path,
        "--threshold",
        10,
        "--out",
        cli_out,
        sample.parquet,
    )
    # Threads write their shards in whatever order they finish, so the rows are
    # compared as a set rather than the files as bytes.
    assert _sorted(cli_out).equals(_sorted(sample.predictions))
    assert f"{sample.predict_report.predictions:,}" in result.stdout


def test_cluster_parity(sample, tmp_path: Path) -> None:
    cli_out = tmp_path / "clusters.csv"
    _run(
        "cluster",
        "--schema",
        sample.schema_path,
        "--predictions",
        sample.predictions,
        "--out",
        cli_out,
        sample.parquet,
    )
    py_out = tmp_path / "clusters_py.csv"
    sample.linker.cluster(sample.predictions, out=py_out)
    assert cli_out.read_text() == py_out.read_text()


def test_explain_parity(sample) -> None:
    table = pq.read_table(str(sample.predictions))
    row = table.slice(0, 1).to_pylist()[0]
    result = _run(
        "explain",
        "--schema",
        sample.schema_path,
        "--model",
        sample.model_path,
        "--pair",
        f"{row['id_a']},{row['id_b']}",
        sample.parquet,
    )
    explanation = sample.linker.explain(row["id_a"], row["id_b"], model=sample.model)
    assert result.stdout == explanation.text
    as_json = _run(
        "explain",
        "--schema",
        sample.schema_path,
        "--model",
        sample.model_path,
        "--pair",
        f"{row['id_a']},{row['id_b']}",
        "--json",
        sample.parquet,
    )
    assert as_json.stdout.strip() == explanation.json()


def test_report_parity(sample) -> None:
    """The diagnostic commands print the same tables the reports carry."""
    common = ["--schema", sample.schema_path, sample.parquet]
    assert _run("explain-blocking", *common).stdout == sample.linker.explain_blocking().text
    recall = sample.linker.recall(sample.truth)
    assert _run("recall", "--truth", sample.truth, *common).stdout == recall.text
    assert (
        _run("recall", "--truth", sample.truth, "--json", *common).stdout
        == recall.json()
    )
    # `completeness` and `inspect` print timings, so their tables are compared
    # through the JSON and the fields rather than the text.
    completeness = sample.linker.completeness(sample.model)
    cli = json.loads(
        _run("completeness", "--model", sample.model_path, "--json", *common).stdout
    )
    ours = json.loads(completeness.json())
    for key in ("pc_estimate", "pc_bound", "pc_product", "pc_independence"):
        assert cli[key] == ours[key], key
    inspect = _run("inspect", *common).stdout
    assert f"{sample.linker.records:,}" in inspect
