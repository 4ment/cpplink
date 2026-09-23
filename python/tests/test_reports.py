# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Every report prints, and `to_dict` sees every field the binding exposes.

The counts below are recorded on purpose: a C++ struct gaining a field that
the binding forgot leaves the count unchanged, while a field added to the
binding raises it, so the number is bumped here when the addition is meant.
"""

from __future__ import annotations

import json

import pytest
from conftest import CORE_PARQUET

import cpplink
from cpplink import to_dict

FIELDS = {
    "Inspection": 3,
    "LoadStats": 4,
    "MemoryReport": 2,
    "Model": 5,
    "ModelComparison": 5,
    "ModelLevel": 7,
    "EstimateReport": 11,
    "SessionReport": 19,
    "SessionFit": 14,
    "PairResidual": 9,
    "InteractionReport": 8,
    "PredictReport": 21,
    "RescoreReport": 13,
    "SpillManifest": 8,
    "ClusterResult": 4,
    "ClusterAssignment": 7,
    "ClusterReport": 19,
    "ClusterQuality": 9,
    "MergeReport": 10,
    "Explanation": 8,
    "SearchResult": 3,
    "SearchReport": 14,
    "SearchHit": 8,
    "PairWaterfall": 16,
    "WaterfallStep": 11,
    "BlockingReport": 7,
    "SourceSummary": 6,
    "RecallResult": 3,
    "RecallMetrics": 12,
    "MissReport": 13,
    "ProfileReport": 37,
    "ColumnProfile": 13,
    "LevelsReport": 10,
    "ComparisonLevels": 29,
    "SimplifyReport": 13,
    "ComparisonSimplify": 8,
    "CompletenessReport": 41,
    "DraftReport": 4,
}


def _first_surname(sample) -> str:
    """A value the store holds, so the search finds something."""
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(str(sample.parquet), columns=["last_name"])
    return str(table.column("last_name")[0].as_py())


@pytest.fixture(scope="module")
def reports(sample, tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("reports")
    linker = sample.linker
    model = sample.model
    spill = root / "spill"
    linker.predict(model, root / "p.parquet", threshold=10, spill=spill)
    predict = linker.last_predict
    shards = root / "shards"
    linker.predict(model, shards, threshold=10, threads=2)
    linker.cluster(sample.predictions, truth=sample.truth)
    cluster = linker.last_cluster
    explanation = linker.explain(linker.id_of(0), linker.id_of(1), model=model)
    hits = linker.search({"last_name": _first_surname(sample)}, model, k=3)
    recall = linker.recall(sample.truth, why=True)
    levels, _ = linker.levels()
    simplify, _ = linker.simplify(model)
    inspection = linker.inspect()
    linker.rescore(model, spill, root / "r.parquet", threshold=12)
    rescore = linker.last_rescore
    reports = {
        "Inspection": inspection,
        "LoadStats": inspection.stats,
        "MemoryReport": inspection.memory,
        "Model": model,
        "ModelComparison": model.comparisons[0],
        "ModelLevel": model.comparisons[0].levels[0],
        "EstimateReport": sample.estimate_report,
        "SessionReport": sample.estimate_report.sessions[0],
        "SessionFit": sample.estimate_report.sessions[0].fit,
        "PairResidual": sample.estimate_report.sessions[0].fit.residuals[0],
        "InteractionReport": sample.estimate_report.interactions,
        "PredictReport": predict,
        "RescoreReport": rescore,
        "SpillManifest": rescore.manifest,
        "ClusterResult": cluster,
        "ClusterAssignment": cluster.assignment,
        "ClusterReport": cluster.report,
        "ClusterQuality": cluster.quality,
        "Explanation": explanation,
        "SearchResult": hits,
        "SearchReport": hits.report,
        "SearchHit": hits.hits[0],
        "PairWaterfall": explanation.waterfall,
        "WaterfallStep": explanation.waterfall.steps[0],
        "BlockingReport": linker.explain_blocking(),
        "SourceSummary": linker.explain_blocking().sources[0],
        "RecallResult": recall,
        "RecallMetrics": recall.metrics,
        "MissReport": recall.misses,
        "ProfileReport": linker.profile(),
        "ColumnProfile": linker.profile(pairs=False, anchors=False).columns[0],
        "LevelsReport": levels,
        "ComparisonLevels": levels.comparisons[0],
        "SimplifyReport": simplify,
        "ComparisonSimplify": simplify.comparisons[0],
        "CompletenessReport": linker.completeness(model),
        "DraftReport": sample.draft,
    }
    if CORE_PARQUET:
        # Naming binary shards' rows needs the file's ids, which the core loads.
        reports["MergeReport"] = cpplink.merge_predictions(
            shards, root / "merged.csv", schema=sample.schema_path, files=[sample.parquet]
        )
    return reports


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_to_dict_sees_every_field(reports, name: str) -> None:
    if name not in reports:
        pytest.skip("this build reads and writes parquet through pandas only")
    report = reports[name]
    assert type(report).__name__ == name
    as_dict = to_dict(report)
    assert isinstance(as_dict, dict)
    assert len(as_dict) == FIELDS[name], sorted(as_dict)
    # Plain Python all the way down: what `json.dumps` accepts.
    json.dumps(as_dict)


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_repr_is_not_empty(reports, name: str) -> None:
    if name not in reports:
        pytest.skip("this build reads and writes parquet through pandas only")
    assert repr(reports[name]).strip()


def test_text_is_the_repr(reports) -> None:
    for name in ("PredictReport", "ClusterReport", "ProfileReport", "RecallMetrics"):
        report = reports[name]
        assert report.text == repr(report)


def test_to_dict_passes_plain_values_through() -> None:
    assert to_dict(3) == 3
    assert to_dict("x") == "x"
    assert to_dict([1, (2, 3)]) == [1, [2, 3]]
    assert to_dict({"a": None}) == {"a": None}


def test_options_are_read_write() -> None:
    options = cpplink.EstimateOptions()
    assert options.seed == 20260903
    options.seed = 1
    options.interactions.enabled = True
    options.ball.budget = 10
    assert (options.seed, options.interactions.enabled, options.ball.budget) == (
        1,
        True,
        10,
    )
    for cls in (
        cpplink.ProfileOptions,
        cpplink.LevelsOptions,
        cpplink.SimplifyOptions,
        cpplink.CompletenessOptions,
    ):
        cls()


def test_estimate_full_text_lists_every_residual_pair(sample) -> None:
    """`text` is the terminal's compact report and `full_text` is `--report`."""
    report = sample.estimate_report
    assert "worst pair" in report.text
    assert "residuals" not in report.text
    assert "residuals" in report.full_text
    assert "worst pair" not in report.full_text
    listed = sum(len(session.fit.residuals) for session in report.sessions)
    assert listed > 0
    rows = [
        line
        for line in report.full_text.splitlines()
        if line.startswith("             ")
        and not line.lstrip().startswith(("Comparison", "patterns"))
        and " patterns over " not in line
    ]
    assert len(rows) == listed
