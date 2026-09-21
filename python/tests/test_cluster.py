# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The numpy views over the partition agree with the file `cluster` writes."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import ROWS


def test_views_are_one_entry_per_record(sample, tmp_path: Path) -> None:
    sample.linker.cluster(sample.predictions, out=tmp_path / "c.csv")
    result = sample.linker.last_cluster
    root = result.assignment.root
    size = result.assignment.size
    assert isinstance(root, np.ndarray) and root.dtype == np.uint32
    assert root.shape == (ROWS,) and size.shape == (ROWS,)
    assert len(result.assignment) == ROWS
    assert not root.flags.writeable
    with pytest.raises(ValueError):
        root[0] = 1
    # A root is its own representative, and only roots carry a size.
    roots = np.unique(root)
    assert (root[roots] == roots).all()
    assert (size[roots] > 0).all()
    assert size[np.setdiff1d(np.arange(ROWS, dtype=np.uint32), roots)].sum() == 0
    assert size.sum() == ROWS
    assert int((size >= 2).sum()) == result.assignment.clusters == result.report.clusters
    assert int((size == 1).sum()) == result.assignment.singletons
    assert result.assignment.largest == int(size.max())
    a, b = int(roots[0]), int(np.flatnonzero(root == roots[0])[-1])
    assert result.assignment.same_cluster(a, b)


def test_views_agree_with_the_written_clusters(sample, tmp_path: Path) -> None:
    out = tmp_path / "clusters.csv"
    sample.linker.cluster(sample.predictions, out=out)
    result = sample.linker.last_cluster
    with open(out, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.report.written
    linker = sample.linker
    by_cluster: Counter[str] = Counter()
    for row in rows:
        record = linker.row_of(row["unique_id"])
        representative = linker.row_of(row["cluster_id"])
        assert result.assignment.root[record] == representative
        assert int(row["cluster_size"]) == result.assignment.size[representative]
        by_cluster[row["cluster_id"]] += 1
    assert len(by_cluster) == result.report.clusters
    assert all(
        count == result.assignment.size[linker.row_of(cluster)]
        for cluster, count in by_cluster.items()
    )


def test_view_outlives_the_result(sample) -> None:
    sample.linker.cluster(sample.predictions)
    root = sample.linker.last_cluster.assignment.root
    import gc

    gc.collect()
    assert root.shape == (ROWS,)
    assert int(root.max()) < ROWS


def test_threshold_reclusters_without_rescoring(sample) -> None:
    sample.linker.cluster(sample.predictions)
    low = sample.linker.last_cluster
    sample.linker.cluster(sample.predictions, threshold=60)
    high = sample.linker.last_cluster
    assert high.report.predictions_used < low.report.predictions_used
    assert high.report.clusters <= low.report.clusters
    assert high.report.predictions_read == low.report.predictions_read
