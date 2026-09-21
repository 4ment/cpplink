# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""`cpplink_viewer` serves what a run wrote, and where it reimplements the core
(the union-find, the shard format) it agrees with it."""

from __future__ import annotations

import csv
import json
import threading
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import ROWS

import cpplink

duckdb = pytest.importorskip("duckdb")

from cpplink_viewer import Options, Viewer, make_server  # noqa: E402
from cpplink_viewer.cache import read_shard_edges  # noqa: E402
from cpplink_viewer.inputs import EDGE_DTYPE, EDGE_MAGIC  # noqa: E402

THRESHOLD = 10  # what conftest's predict wrote at
HIGHER = 60  # a threshold that prunes a good share of the fixture's predictions


class Run:
    """A clustered run with its waterfalls, and the viewer's options over it."""

    def __init__(self, sample, root: Path) -> None:
        self.sample = sample
        self.clusters = root / "clusters.csv"
        self.cluster_result = sample.linker.cluster(sample.predictions, out=self.clusters)
        self.waterfalls = root / "waterfalls.parquet"
        result = cpplink.run(
            [
                "explain",
                "--schema",
                str(sample.schema_path),
                "--model",
                str(sample.model_path),
                "--predictions",
                str(sample.predictions),
                "--out",
                str(self.waterfalls),
                str(sample.parquet),
            ]
        )
        assert result.code == 0, result.stderr
        self.shards = root / "shards"
        sample.linker.predict(sample.model, self.shards, threshold=THRESHOLD)
        self.root = root

    def options(self, **overrides) -> Options:
        opts = Options(
            data=[str(self.sample.parquet)],
            schema=str(self.sample.schema_path),
            clusters=str(self.clusters),
            predictions=str(self.sample.predictions),
            truth=str(self.sample.truth),
            waterfalls=str(self.waterfalls),
            model=str(self.sample.model_path),
            cache=str(self.root / "cache.duckdb"),
        )
        for key, value in overrides.items():
            setattr(opts, key, value)
        return opts

    def written_clusters(self) -> dict[str, str]:
        with open(self.clusters, newline="") as handle:
            return {r["unique_id"]: r["cluster_id"] for r in csv.DictReader(handle)}


@pytest.fixture(scope="module")
def run(sample, tmp_path_factory: pytest.TempPathFactory) -> Run:
    return Run(sample, tmp_path_factory.mktemp("viewer"))


@pytest.fixture(scope="module")
def viewer(run: Run) -> Viewer:
    return Viewer.open(run.options(), log=lambda _: None)


def test_summary_counts_the_cluster_file(run: Run, viewer: Viewer) -> None:
    summary = viewer.summary()
    written = run.written_clusters()
    assert summary["totals"]["clusters"] == len(set(written.values()))
    assert summary["totals"]["records"] == len(written)
    assert summary["totals"]["pairs"] == pq.read_metadata(run.sample.predictions).num_rows
    with open(run.sample.schema_path) as handle:
        declared = json.load(handle)["columns"]
    assert summary["columns"] == [c["name"] for c in declared if "derive" not in c]
    assert summary["has_truth"] and summary["has_model"]


def test_listing_pages_through_every_cluster(viewer: Viewer) -> None:
    total = viewer.summary()["totals"]["clusters"]
    seen = []
    offset = 0
    while True:
        page = viewer.listing("", "any", "", "size", True, "all", offset, 7)
        assert page["matched"] == total
        if not page["clusters"]:
            break
        seen.extend(c["id"] for c in page["clusters"])
        offset += len(page["clusters"])
    assert len(seen) == total and len(set(seen)) == total
    # The filters are subsets, and an id search names the record it found.
    split = viewer.listing("", "any", "", "discord", True, "split", 0, 500)
    assert 0 < split["matched"] <= total
    assert all(c["discord"] > 0 for c in split["clusters"])
    first = viewer.cluster(seen[0])
    uid = first["rows"][0]["id"]
    by_id = viewer.listing(uid + ", no-such-id", "id", "", "discord", True, "all", 0, 10)
    assert by_id["matched"] == 1 and by_id["found"] == [uid]
    assert by_id["clusters"][0]["id"] == seen[0]
    with pytest.raises(ValueError, match="no column"):
        viewer.listing("x", "col", "no_such_column", "discord", True, "all", 0, 10)


def test_cluster_holds_its_members_and_edges(run: Run, viewer: Viewer) -> None:
    written = run.written_clusters()
    members = defaultdict(set)
    for uid, cid in written.items():
        members[cid].add(uid)
    largest = max(members, key=lambda c: len(members[c]))
    cluster = viewer.cluster(largest)
    assert cluster["size"] == len(members[largest])
    assert {r["id"] for r in cluster["rows"]} == members[largest]
    assert len(cluster["dist"]) == len(viewer.columns)
    assert all(len(r["v"]) == len(viewer.columns) for r in cluster["rows"])
    assert cluster["edge_count"] == len(cluster["edges"]) >= len(members[largest]) - 1
    assert all(
        a in members[largest] and b in members[largest] for a, b, _ in cluster["edges"]
    )
    assert cluster["weakest"] == pytest.approx(
        min(w for _, _, w in cluster["edges"]), abs=1e-3
    )
    # With truth, every member carries an entity label.
    assert all("t" in r for r in cluster["rows"])
    assert cluster["entities"] >= 1
    assert viewer.cluster("no-such-cluster") is None


def test_pair_is_the_scorer_ledger_read_back(run: Run, viewer: Viewer) -> None:
    table = pq.read_table(
        run.sample.predictions, columns=["id_a", "id_b", "match_weight"]
    )
    row = table.slice(0, 1).to_pylist()[0]
    pair = viewer.pair(row["id_a"], row["id_b"])
    assert pair is not None
    assert pair["w"] == pytest.approx(row["match_weight"], abs=1e-3)
    ledger = pair["waterfall"]
    assert ledger["weight"] == pytest.approx(row["match_weight"], abs=1e-9)
    # The ledger is the explain report's: same steps, and the running total
    # ends at the weight.
    explanation = run.sample.linker.explain(
        row["id_a"], row["id_b"], model=run.sample.model
    )
    assert [s["name"] for s in ledger["steps"]] == run.sample.schema.comparison_names
    assert ledger["steps"][-1]["running"] + sum(
        i["bits"] for i in ledger["interactions"]
    ) == pytest.approx(ledger["weight"], abs=1e-9)
    assert ledger["prior"] == pytest.approx(explanation.waterfall.prior, abs=1e-9)
    assert {r["id"] for r in pair["rows"]} == {row["id_a"], row["id_b"]}
    assert viewer.pair("no", "such") is None


def test_threshold_reclusters_as_the_core_does(run: Run) -> None:
    """The Python union-find, over the file and over the shards, produces the
    partition `cluster --threshold` writes, named by the same representatives."""
    for predictions in (run.sample.predictions, run.shards):
        # Each source is clustered by the core itself: the shards are a second
        # run's, so their edge order, and with it the representatives, is its own.
        expected_path = run.root / f"clusters_{HIGHER}_{predictions.name}.csv"
        run.sample.linker.cluster(predictions, threshold=HIGHER, out=expected_path)
        with open(expected_path, newline="") as handle:
            expected = {r["unique_id"]: r["cluster_id"] for r in csv.DictReader(handle)}
        assert expected  # the fixture must leave something above the higher threshold
        opts = run.options(
            clusters=None,
            predictions=str(predictions),
            threshold=HIGHER,
            cache=str(run.root / f"cache_{predictions.name}.duckdb"),
        )
        viewer = Viewer.open(opts, log=lambda _: None)
        got = dict(viewer.query("SELECT uid, cluster_id FROM records"))
        assert got == expected
        # Clustered at the threshold, no prediction crosses two clusters.
        assert viewer.query(
            "SELECT count(*) FROM pairs WHERE cluster_a <> cluster_b"
        ) == [(0,)]
        viewer.conn.close()


def test_a_prediction_across_two_clusters_is_rejected_under_both(run: Run) -> None:
    """A prediction clustering overruled is listed under both of its clusters,
    naming the other. The fixture has none, so one is added: the run's
    predictions as csv plus a pair between two clusters."""
    written = run.written_clusters()
    first, second = sorted(set(written.values()))[:2]
    a = min(u for u, c in written.items() if c == first)
    b = min(u for u, c in written.items() if c == second)
    table = pq.read_table(
        run.sample.predictions, columns=["id_a", "id_b", "match_weight"]
    )
    predictions = run.root / "with_crossing.csv"
    with open(predictions, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["id_a", "id_b", "match_weight"])
        out.writerows(list(r.values()) for r in table.to_pylist())
        out.writerow([a, b, 20.0])
    opts = run.options(
        predictions=str(predictions),
        waterfalls=None,
        model=None,
        cache=str(run.root / "cache_rejected.duckdb"),
    )
    viewer = Viewer.open(opts, log=lambda _: None)
    assert viewer.summary()["totals"]["pairs"] == table.num_rows + 1
    assert viewer.query("SELECT a, b FROM pairs WHERE cluster_a <> cluster_b") == [(a, b)]
    for mine, other in ((first, second), (second, first)):
        rejected = viewer.cluster(mine)["rejected"]
        assert [(r["a"], r["b"], r["w"], r["other"]) for r in rejected] == [
            (a, b, 20.0, other)
        ]
        assert rejected[0]["same"] is False  # two entities, by the truth file
    pair = viewer.pair(a, b)
    assert (pair["ca"], pair["cb"]) == (first, second) and pair["waterfall"] is None
    # Filtering at a threshold above it drops it from the cache altogether.
    opts.threshold = 30.0
    opts.cache = str(run.root / "cache_rejected_30.duckdb")
    viewer = Viewer.open(opts, log=lambda _: None)
    assert viewer.query("SELECT count(*) FROM pairs WHERE cluster_a <> cluster_b") == [
        (0,)
    ]
    assert viewer.summary()["totals"]["pairs"] == table.num_rows


def test_shards_are_read_as_the_core_wrote_them(run: Run) -> None:
    shards = sorted(run.shards.glob("shard-*.bin"))
    assert shards
    with open(shards[0], "rb") as handle:
        assert handle.read(len(EDGE_MAGIC)) == EDGE_MAGIC
    table = read_shard_edges(str(run.shards), THRESHOLD)
    merged = pq.read_table(run.sample.predictions, columns=["match_weight"])
    assert table.num_rows == merged.num_rows
    assert sorted(table["w"].to_pylist()) == pytest.approx(
        sorted(merged["match_weight"].to_pylist()), abs=1e-9
    )
    assert EDGE_DTYPE.itemsize == 20
    assert np.asarray(table["a"]).max() < ROWS


def test_cache_is_reused_until_an_input_changes(run: Run) -> None:
    opts = run.options(cache=str(run.root / "cache_reuse.duckdb"))
    said = []
    Viewer.open(opts, log=said.append).conn.close()
    assert any("building" in line for line in said)
    said.clear()
    Viewer.open(opts, log=said.append).conn.close()
    assert said == []
    opts.max_rows += 1  # part of the fingerprint
    Viewer.open(opts, log=said.append).conn.close()
    assert any("building" in line for line in said)


def test_refused_options_say_why(run: Run) -> None:
    assert Options(data=["x"], schema="s").check() == (
        "--clusters is needed without --threshold"
    )
    assert (
        "go together" in Options(data=["x"], schema="s", clusters="c", model="m").check()
    )
    assert (
        "needs --predictions"
        in Options(data=["x"], schema="s", clusters="c", threshold=1.0).check()
    )
    assert run.options().check() is None
    with pytest.raises(SystemExit, match="--clusters is needed"):
        Viewer.open(Options(data=["x"], schema="s"))


def test_http_serves_the_page_and_the_api(viewer: Viewer) -> None:
    server = make_server(viewer, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        where = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(where + "/") as answer:
            page = answer.read().decode()
        assert "<title>cpplink clusters</title>" in page
        assert "__CLUSTER_DATA__" not in page
        with urlopen(where + "/api/summary") as answer:
            assert json.load(answer) == viewer.summary()
        with urlopen(where + "/api/list?sort=size&limit=3") as answer:
            listing = json.load(answer)
        assert len(listing["clusters"]) == 3
        first = listing["clusters"][0]["id"]
        with urlopen(where + f"/api/cluster?id={first}") as answer:
            cluster = json.load(answer)
        assert cluster["id"] == first
        edge = cluster["edges"][0]
        with urlopen(where + f"/api/pair?a={edge[0]}&b={edge[1]}") as answer:
            pair = json.load(answer)
        assert pair["w"] == edge[2]
        with pytest.raises(Exception, match="404"):
            urlopen(where + "/api/cluster?id=no-such-cluster")
        with pytest.raises(Exception, match="404"):
            urlopen(where + "/nothing")
    finally:
        server.shutdown()
        server.server_close()
