# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The Linker over the sample fixture: every stage, and link mode."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from conftest import ROWS

import cpplink


def test_linker_loads_the_store(sample) -> None:
    linker = sample.linker
    assert linker.records == ROWS
    assert linker.datasets == 1
    assert linker.mode == "all pairs"
    assert not linker.all_pairs
    assert linker.stats.rows == ROWS
    assert "records" in repr(linker)
    inspection = linker.inspect()
    assert inspection.stats.rows == ROWS
    assert inspection.memory.total > 0
    assert any(line.structure for line in inspection.memory.lines)
    assert "Records" in inspection.text
    assert linker.row_of(linker.id_of(5)) == 5


def test_linker_takes_a_schema_path(sample) -> None:
    linker = cpplink.Linker(sample.schema_path, [sample.parquet])
    assert linker.records == ROWS


def test_bad_mode_is_refused(sample) -> None:
    with pytest.raises(cpplink.Error, match="--mode must be"):
        cpplink.Linker(sample.schema, sample.parquet, mode="sideways")


def test_missing_file_is_refused(sample) -> None:
    with pytest.raises(cpplink.Error):
        cpplink.Linker(sample.schema, "/nonexistent.parquet")


def test_estimate_returns_a_model_and_a_report(sample) -> None:
    model, report = sample.model, sample.estimate_report
    assert model.records == ROWS
    assert 0 < model.lambda_ < 1
    assert model.prior_weight() < 0
    names = [comparison.name for comparison in model.comparisons]
    assert names == sample.schema.comparison_names
    level = model.comparisons[0].levels[0]
    assert level.label
    assert level.weight() == pytest.approx(
        __import__("math").log2(level.m / level.u), rel=1e-9
    )
    assert report.sessions
    assert all(session.column for session in report.sessions)
    assert report.u_pairs > 0
    assert "lambda" in model.text
    assert "Session" in report.text or "session" in report.text
    assert sample.model_path.exists()
    assert cpplink.Model.from_file(str(sample.model_path)).to_json() == model.to_json()
    assert cpplink.Model.from_json(model.to_json()).lambda_ == model.lambda_


def test_estimate_refuses_unknown_option(sample) -> None:
    with pytest.raises(TypeError, match="unknown option"):
        sample.linker.estimate(options=cpplink._apply(cpplink.EstimateOptions(), nope=1))


def test_predict_writes_one_file(sample) -> None:
    report = sample.predict_report
    assert report.merged_path == str(sample.predictions)
    assert report.predictions > 0
    assert report.enumerated > report.predictions
    assert report.mode == "all pairs"
    assert "Candidate" in report.text or "candidate" in report.text
    table = pq.read_table(str(sample.predictions))
    assert table.num_rows == report.predictions
    assert table.schema.names == [
        "id_a",
        "id_b",
        "gamma",
        "match_weight",
        "match_probability",
    ]
    assert min(table["match_weight"].to_pylist()) >= 10


def test_predict_needs_a_threshold(sample, tmp_path: Path) -> None:
    with pytest.raises(cpplink.Error, match="threshold"):
        sample.linker.predict(sample.model, tmp_path / "x.parquet")
    with pytest.raises(cpplink.Error, match="format"):
        sample.linker.predict(
            sample.model, tmp_path / "x.parquet", threshold=10, format="csv"
        )


def test_predict_to_shards_and_merge(sample, tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    report = sample.linker.predict(sample.model, shards, threshold=10, threads=2)
    assert report.merged_path == ""
    assert len(report.shards) == 2
    merged = tmp_path / "merged.csv"
    merge = cpplink.merge_predictions(
        shards, merged, schema=sample.schema_path, files=[sample.parquet]
    )
    assert merge.written == sample.predict_report.predictions
    assert merge.format == "csv"
    assert not merge.row_indices
    assert "Predictions" in merge.text or "prediction" in merge.text.lower()
    lines = merged.read_text().splitlines()
    assert lines[0].startswith("id_a,id_b")
    assert len(lines) == merge.written + 1


def test_cluster_scores_the_planted_pairs(sample, tmp_path: Path) -> None:
    out = tmp_path / "clusters.csv"
    result = sample.linker.cluster(sample.predictions, truth=sample.truth, out=out)
    assert result.quality is not None
    # Not exactly 1.0: `gen_sample` draws through the standard library's
    # distributions, whose output differs between libstdc++ and libc++, so the
    # fixture is a different file on Linux and there one planted pair sits under
    # the threshold (F1 0.9992 against 1.0 on macOS). The bound is the link
    # fixture's, below.
    assert result.quality.f1 > 0.99
    assert result.quality.precision > 0.99
    assert result.report.records == ROWS
    assert result.report.predictions_read == sample.predict_report.predictions
    assert result.report.written > 0
    assert out.exists()
    assert "F1" in result.text or "f1" in result.text.lower()
    without = sample.linker.cluster(sample.predictions)
    assert without.quality is None
    assert without.report.clusters == result.report.clusters


def test_cluster_file_matches_the_linker(sample) -> None:
    light = cpplink.cluster_file(sample.schema_path, [sample.parquet], sample.predictions)
    full = sample.linker.cluster(sample.predictions)
    assert light.report.clusters == full.report.clusters
    assert (light.assignment.root == full.assignment.root).all()


def test_explain_agrees_with_the_prediction(sample) -> None:
    table = pq.read_table(str(sample.predictions))
    row = table.slice(0, 1).to_pylist()[0]
    explanation = sample.linker.explain(row["id_a"], row["id_b"], model=sample.model)
    assert explanation.weight == pytest.approx(row["match_weight"], abs=1e-9)
    assert explanation.waterfall.gamma == row["gamma"]
    assert set(explanation.levels) == set(sample.schema.comparison_names)
    assert all(explanation.levels.values())
    assert explanation.id_a == row["id_a"]
    assert '"weight"' in explanation.json()
    assert "prior" in explanation.text.lower()
    by_row = sample.linker.explain(
        explanation.row_a, explanation.row_b, by_row=True, model=sample.model
    )
    assert by_row.weight == explanation.weight
    plain = sample.linker.explain(row["id_a"], row["id_b"])
    assert plain.weight is None and plain.waterfall is None
    assert plain.levels == explanation.levels
    with pytest.raises(cpplink.Error, match="json needs a model"):
        plain.json()
    with pytest.raises(cpplink.Error, match="no record with id"):
        sample.linker.explain("no-such-id", row["id_b"])


def test_explain_blocking_prices_every_source(sample) -> None:
    report = sample.linker.explain_blocking(count=True)
    assert report.records == ROWS
    assert len(report.sources) == len(sample.schema.blocking)
    assert report.candidate_sum == sum(source.pairs for source in report.sources)
    assert report.counted_union
    assert 0 < report.candidate_union <= report.candidate_sum
    assert report.candidate_union == sample.predict_report.enumerated
    assert "Source" in report.text


def test_recall_and_misses(sample) -> None:
    result = sample.linker.recall(sample.truth, why=True, show_misses=3)
    assert result.metrics.truth_pairs > 0
    assert result.metrics.pair_completeness > 0.99
    assert len(result.metrics.sources) == len(sample.schema.blocking)
    assert result.misses is not None
    assert result.misses.missed == result.metrics.truth_pairs - result.metrics.union_found
    assert len(result.misses.reasons) == len(sample.schema.blocking)
    assert cpplink.MissReport.reason_names[0] == "produced"
    assert '"truth_pairs"' in result.json()
    assert "Pair completeness" in result.text or "PC" in result.text
    plain = sample.linker.recall(sample.truth)
    assert plain.misses is None
    assert plain.miss_text == ""


def test_profile(sample) -> None:
    report = sample.linker.profile(truth=sample.truth)
    assert report.records == ROWS
    assert report.columns
    assert report.margin_bits > 0
    assert report.truthed
    assert '"margin_bits"' in report.json() or '"margin' in report.json()
    assert "margin" in report.text.lower()
    quick = sample.linker.profile(pairs=False, anchors=False)
    assert not quick.walked
    assert not quick.anchored


def test_levels_writes_a_schema_that_loads(sample, tmp_path: Path) -> None:
    out = tmp_path / "levels.json"
    report, proposal = sample.linker.levels(out=out)
    assert report.records == ROWS
    assert report.comparisons
    assert proposal.comparison_names == sample.schema.comparison_names
    assert out.exists()
    cpplink.Linker(proposal, sample.parquet)
    assert "levels" in report.text.lower()
    assert report.json().startswith("{")


def test_simplify(sample, tmp_path: Path) -> None:
    report, proposal = sample.linker.simplify(sample.model, min_gap=1.0)
    assert report.records == ROWS
    assert len(report.comparisons) == len(sample.schema.comparison_names)
    if report.merged_levels:
        assert proposal is not None
        assert proposal.gamma_width < sample.schema.gamma_width
        cpplink.Linker(proposal, sample.parquet)
    else:
        assert proposal is None
    assert report.json().startswith("{")
    with pytest.raises(cpplink.Error, match="alpha"):
        sample.linker.simplify(sample.model, alpha=2.0)


def test_completeness(sample) -> None:
    report = sample.linker.completeness(sample.model, truth=sample.truth)
    assert report.records == ROWS
    assert report.measured
    assert 0.9 < report.pc_measured <= 1.0
    assert 0.9 < report.pc_estimate <= 1.0
    assert report.pc_basis
    assert '"pc_estimate"' in report.json()
    assert "completeness" in report.text.lower()


def test_rescore_replays_a_spill(sample, tmp_path: Path) -> None:
    spill = tmp_path / "spill"
    first = sample.linker.predict(
        sample.model, tmp_path / "first.parquet", threshold=10, spill=spill
    )
    assert first.spilled > 0
    report = sample.linker.rescore(
        sample.model, spill, tmp_path / "again.parquet", threshold=20
    )
    assert report.pairs == first.spilled
    assert report.predictions <= first.predictions
    assert not report.below_spill_threshold
    assert report.manifest.threshold == 10
    again = pq.read_table(str(tmp_path / "again.parquet"))
    assert again.num_rows == report.predictions
    assert min(again["match_weight"].to_pylist()) >= 20
    lower = sample.linker.rescore(
        sample.model, spill, tmp_path / "lower.parquet", threshold=5
    )
    assert lower.below_spill_threshold
    assert "below" in lower.text.lower()


def test_all_pairs(sample, tmp_path: Path) -> None:
    unblocked = cpplink.Linker(sample.schema, sample.parquet, all_pairs=True)
    assert unblocked.all_pairs
    report = unblocked.explain_blocking()
    assert [source.kind for source in report.sources] == ["all_pairs"]
    assert report.sources[0].pairs == ROWS * (ROWS - 1) // 2


def test_link_mode(link, tmp_path: Path) -> None:
    linker = link.linker
    assert linker.datasets == 2
    assert linker.mode == "cross-dataset pairs"
    recall = linker.recall(link.truth)
    assert recall.metrics.pair_completeness > 0.99
    model, _ = linker.estimate(seed=3)
    out = tmp_path / "link.parquet"
    predict = linker.predict(model, out, threshold=10)
    assert predict.datasets == 2
    table = pq.read_table(str(out))
    assert table.schema.names[:4] == ["dataset_a", "id_a", "dataset_b", "id_b"]
    assert set(table["dataset_a"].to_pylist()) == {"a"}
    assert set(table["dataset_b"].to_pylist()) == {"b"}
    result = linker.cluster(out, truth=link.truth)
    assert result.quality.f1 > 0.99
    row = table.slice(0, 1).to_pylist()[0]
    explanation = linker.explain(f"a:{row['id_a']}", f"b:{row['id_b']}", model=model)
    assert explanation.weight == pytest.approx(row["match_weight"], abs=1e-9)
    both = cpplink.Linker(link.schema, [link.a, link.b], mode="link-and-dedup")
    assert both.mode == "all pairs"
