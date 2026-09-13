#!/usr/bin/env python3
# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Build a self-contained HTML viewer for the clusters a cpplink run produced.

    cpplink predict --schema s.json --model m.json --out predictions.parquet data.parquet
    cpplink cluster --schema s.json --predictions predictions.parquet \
        --out clusters.csv data.parquet
    cpplink explain --schema s.json --model m.json --predictions predictions.parquet \
        --out waterfalls.parquet data.parquet
    tools/cluster_view.py --schema s.json --clusters clusters.csv \
        --predictions predictions.parquet --waterfalls waterfalls.parquet \
        --model m.json --out clusters.html data.parquet

Reads the cluster assignment, pulls each member's column values back out of the
parquet, and writes one HTML file holding the clusters it selected. Open it in a
browser: the left pane lists clusters, the right one shows the members side by
side with every disagreeing cell highlighted, so what a cluster has in common is
the part that is not highlighted.

The `network` checkbox in the header draws the selected cluster's predictions as
a graph, which is where a chain shows itself as a chain. It is off by default and
the choice is remembered, because the layout is the one quadratic thing the page
does and most clusters are read without it.

With `--waterfalls`, every embedded prediction also carries its waterfall: the
prior, then what each comparison charged and its term-frequency move, ending at
the match weight. The page lists the pairs beside the clusters and draws the
ledger for the one picked. The file is what `cpplink explain --predictions
<file> --out <file>` writes, one wide row per prediction, and nothing here
recomputes a bit of it: the chart is the scorer's own arithmetic, read back. The
level labels and rates come from `--model`, since they are the model's and not
the pair's. Predictions the run made between two records that clustering then
put in different clusters are kept as well, since a pair scored above the write
threshold and below the clustering one is the pair most worth reading.

A run ends with single files, so `--clusters` and `--predictions` each name one
file and its extension picks csv or parquet. `--predictions` still takes the
shard directory too, which names records by row rather than by `unique_id` and so
needs the row index a merged file makes unnecessary.

Nothing here holds a row per record. The clusters are chosen in Arrow, the
parquet is walked one row group at a time and only the row groups holding a
chosen member are materialised, and the predictions are filtered in batches, so
the resident cost follows the clusters embedded rather than the file's size.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

EDGE_MAGIC = b"CPPLNKE1"
EDGE_DTYPE = np.dtype([("a", "<u4"), ("b", "<u4"), ("g", "<u4"), ("w", "<f8")])
EDGE_CHUNK = 1 << 22  # edges read at a time: 80 MB of shard, whatever its size
# A merged prediction file names records by `unique_id`, and the ids are read as
# strings whatever the file holds, because that is what the id column gives.
PREDICTION_IDS = {"id_a": pa.string(), "id_b": pa.string()}
PREDICTION_COLUMNS = ["id_a", "id_b", "match_weight"]
PREDICTION_BATCH = 1 << 20


def schema_columns(path):
    """The id column and the columns actually read from the file."""
    with open(path) as handle:
        schema = json.load(handle)
    id_column = schema.get("unique_id", "unique_id")
    columns = [c["name"] for c in schema.get("columns", []) if "derive" not in c]
    return id_column, columns


def cell(value):
    """One value as the page shows it: a string, a joined list, or nothing."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(cell(v) or "" for v in value) or None
    return str(value)


def as_strings(table, names):
    """The named columns as utf8, whatever the file stored them as."""
    for name in names:
        at = table.column_names.index(name)
        if table.column(name).type != pa.string():
            table = table.set_column(at, name, pc.cast(table.column(name), pa.string()))
    return table


def read_assignment(path):
    """The cluster file as an Arrow table, ids left as strings.

    `cluster --out` names one file, so the extension picks the reader: csv is
    what it writes today and parquet is read the same way.
    """
    if path.endswith(".parquet"):
        table = pq.read_table(path)
    else:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                column_types={"unique_id": pa.string(), "cluster_id": pa.string()}))
    for name in ("unique_id", "cluster_id", "cluster_size"):
        if name not in table.column_names:
            raise SystemExit(f"{path}: not a cpplink cluster file")
    return as_strings(table, ("unique_id", "cluster_id"))


def choose_clusters(table, args):
    """Which clusters to embed, decided over one row per cluster.

    The cluster id is the representative's own unique id, so the rows where the
    two columns are equal are a directory of the clusters with their sizes: one
    Arrow filter instead of grouping the members.
    """
    heads = table.filter(pc.equal(table["unique_id"], table["cluster_id"]))
    size = heads["cluster_size"]
    keep = pc.greater_equal(size, args.min_size)
    if args.max_size:
        keep = pc.and_(keep, pc.less_equal(size, args.max_size))
    heads = heads.filter(keep)

    order = np.arange(heads.num_rows)
    if args.sort == "size":
        order = np.argsort(-heads["cluster_size"].to_numpy(zero_copy_only=False),
                           kind="stable")
    elif args.sort == "id":
        order = pc.sort_indices(heads["cluster_id"]).to_numpy()
    else:
        np.random.default_rng(args.seed).shuffle(order)

    total = heads.num_rows
    if args.limit:
        order = order[: args.limit]
    chosen = heads["cluster_id"].take(pa.array(order)).to_pylist()
    if args.id:
        pinned = [c for c in args.id if c not in set(chosen)]
        chosen = pinned + chosen
    return chosen, total


def read_members(table, chosen):
    """{cluster_id: [unique_id, ...]} for the chosen clusters alone."""
    wanted = pa.array(chosen, type=pa.string())
    rows = table.filter(pc.is_in(table["cluster_id"], value_set=wanted))
    members = {}
    for uid, cid in zip(rows["unique_id"].to_pylist(), rows["cluster_id"].to_pylist()):
        members.setdefault(cid, []).append(uid)
    return members


def read_records(paths, id_column, columns, wanted_ids):
    """Values and store row indices for the wanted ids, a row group at a time.

    A row index in an edge shard is a position in the concatenation of the inputs
    in order, which is how the files were read in, so the offset carried across
    row groups is the same number the shards hold.
    """
    kept, rows_by_id, offset, touched, groups = [], {}, 0, 0, 0
    wanted = pa.array(sorted(wanted_ids), type=pa.string())
    for path in paths:
        handle = pq.ParquetFile(path)
        for group in range(handle.num_row_groups):
            rows = handle.metadata.row_group(group).num_rows
            groups += 1
            ids = handle.read_row_group(group, columns=[id_column]).column(id_column)
            at = pc.indices_nonzero(pc.is_in(ids, value_set=wanted))
            if len(at):
                touched += 1
                block = handle.read_row_group(group, columns=[id_column] + columns)
                kept.append(block.take(at))
                for local, uid in zip(at.to_pylist(),
                                      ids.take(at).to_pylist()):
                    rows_by_id[uid] = offset + local
            offset += rows
    table = pa.concat_tables(kept) if kept else None
    return table, rows_by_id, offset, groups, touched


def read_shard_predictions(directory, rows_by_id):
    """(id_a, id_b) -> weight for the shard edges whose both ends are wanted.

    The shards are the run's whole output, which is the one thing here that can
    be larger than memory, so they are read in fixed chunks and filtered down to
    the embedded rows before anything is kept. A shard names records by row, so
    the row index is what the filter runs over and the ids go back on afterwards.
    """
    found = {}
    names = sorted(n for n in os.listdir(directory)
                   if n.startswith("shard-") and n.endswith(".bin"))
    if not names:
        raise SystemExit(f"{directory}: no shard-*.bin files")
    id_by_row = {row: uid for uid, row in rows_by_id.items()}
    wanted = np.sort(np.asarray(list(id_by_row), dtype=np.uint32))
    read = 0
    for name in names:
        with open(os.path.join(directory, name), "rb") as handle:
            if handle.read(len(EDGE_MAGIC)) != EDGE_MAGIC:
                raise SystemExit(f"{name}: not a cpplink prediction shard")
            while True:
                raw = handle.read(EDGE_CHUNK * EDGE_DTYPE.itemsize)
                if not raw:
                    break
                block = np.frombuffer(raw, dtype=EDGE_DTYPE)
                read += block.size
                keep = block[np.isin(block["a"], wanted) & np.isin(block["b"], wanted)]
                for edge in keep:
                    found[(id_by_row[int(edge["a"])],
                           id_by_row[int(edge["b"])])] = float(edge["w"])
    return found, read


def prediction_batches(path):
    """One merged prediction file, a batch of rows at a time."""
    if path.endswith(".parquet"):
        handle = pq.ParquetFile(path)
        held = handle.schema_arrow.names
        if any(name not in held for name in PREDICTION_COLUMNS):
            raise SystemExit(f"{path}: not a cpplink prediction file")
        for batch in handle.iter_batches(batch_size=PREDICTION_BATCH,
                                         columns=PREDICTION_COLUMNS):
            yield pa.Table.from_batches([batch])
        return
    reader = pacsv.open_csv(
        path, convert_options=pacsv.ConvertOptions(column_types=PREDICTION_IDS))
    if any(name not in reader.schema.names for name in PREDICTION_COLUMNS):
        raise SystemExit(f"{path}: not a cpplink prediction file")
    for batch in reader:
        yield pa.Table.from_batches([batch])


def read_file_predictions(path, wanted_ids):
    """(id_a, id_b) -> weight for the predictions naming two wanted records.

    A merged file names records by `unique_id`, so there is no row index to carry
    and no dependence on which file a record was read from. It is still the run's
    whole output, so it is read a batch at a time and cut to the embedded records
    before anything is kept.
    """
    wanted = pa.array(sorted(wanted_ids), type=pa.string())
    found, read = {}, 0
    for table in prediction_batches(path):
        read += table.num_rows
        table = as_strings(table, ("id_a", "id_b"))
        keep = pc.and_(pc.is_in(table["id_a"], value_set=wanted),
                       pc.is_in(table["id_b"], value_set=wanted))
        table = table.filter(keep)
        for a, b, weight in zip(table["id_a"].to_pylist(), table["id_b"].to_pylist(),
                                table["match_weight"].to_pylist()):
            found[(a, b)] = float(weight)
    return found, read


def read_predictions(path, wanted_ids, rows_by_id):
    """The run's predictions, from the merged file or from the shard directory."""
    if os.path.isdir(path):
        return read_shard_predictions(path, rows_by_id)
    return read_file_predictions(path, wanted_ids)


def read_truth(path, wanted_ids):
    """Group number per wanted id, from the known pairs.

    Being a duplicate is transitive, so the whole file has to be walked: two
    embedded records can be the same entity through a third the page never
    shows. It is walked as integers, though -- both columns are dictionary
    encoded into one dictionary and the union-find runs over that -- so a truth
    file of any size costs one entry per distinct id rather than one string.
    """
    with open(path) as handle:
        first = handle.readline().rstrip("\n").split(",")
    header = first[:1] in (["id_a"], ["unique_id"], ["a"])
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=["a", "b"],
                                       skip_rows=1 if header else 0),
        convert_options=pacsv.ConvertOptions(
            column_types={"a": pa.string(), "b": pa.string()}))
    left = table.column("a").combine_chunks()
    right = table.column("b").combine_chunks()
    pairs = len(left)
    codes = pa.concat_arrays([left, right]).dictionary_encode()
    index = codes.indices.to_numpy(zero_copy_only=False)

    parent = list(range(len(codes.dictionary)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(index[:pairs].tolist(), index[pairs:].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    want = pa.array(sorted(wanted_ids), type=pa.string())
    at = pc.index_in(want, value_set=codes.dictionary).to_pylist()
    return {uid: find(i)
            for uid, i in zip(want.to_pylist(), at) if i is not None}


WATERFALL_FIXED = ["id_a", "id_b", "gamma", "prior", "match_weight", "match_probability",
                   "bracket_low", "bracket_high", "threshold", "zone", "emitted"]


def read_model_levels(path):
    """What a ledger needs from the model file: labels and rates per level.

    A waterfall row names the level each comparison landed on; its label, m and
    u are the model's rather than the pair's, so the file does not repeat them
    and they are read from here once.
    """
    with open(path) as handle:
        model = json.load(handle)
    comparisons = [{"name": c["name"],
                    "levels": [{"label": lvl.get("label", str(i)),
                                "m": lvl.get("m"), "u": lvl.get("u")}
                               for i, lvl in enumerate(c["levels"])]}
                   for c in model["comparisons"]]
    interactions = [f'{t["left"]} x {t["right"]}' for t in model.get("interactions", [])]
    return {"records": model.get("records"), "comparisons": comparisons,
            "interactions": interactions}


def waterfall_batches(path):
    """One waterfall file, a batch of rows at a time, whichever format it is."""
    if path.endswith(".parquet"):
        handle = pq.ParquetFile(path)
        held = handle.schema_arrow.names
        if any(name not in held for name in WATERFALL_FIXED):
            raise SystemExit(f"{path}: not a cpplink waterfall file")
        for batch in handle.iter_batches(batch_size=PREDICTION_BATCH):
            yield pa.Table.from_batches([batch])
        return
    reader = pacsv.open_csv(
        path, convert_options=pacsv.ConvertOptions(column_types=PREDICTION_IDS))
    if any(name not in reader.schema.names for name in WATERFALL_FIXED):
        raise SystemExit(f"{path}: not a cpplink waterfall file")
    for batch in reader:
        yield pa.Table.from_batches([batch])


def read_waterfalls(path, wanted_ids):
    """(id_a, id_b) -> the wide row, for the pairs naming two wanted records."""
    wanted = pa.array(sorted(wanted_ids), type=pa.string())
    found, read = {}, 0
    for table in waterfall_batches(path):
        read += table.num_rows
        table = as_strings(table, ("id_a", "id_b"))
        keep = pc.and_(pc.is_in(table["id_a"], value_set=wanted),
                       pc.is_in(table["id_b"], value_set=wanted))
        for row in table.filter(keep).to_pylist():
            found[(row["id_a"], row["id_b"])] = row
    return found, read


def ledger(row, levels):
    """A wide row expanded to the ledger `cpplink explain --json` writes.

    The same shape for both viewers, so the page has one renderer: the prior,
    one step per comparison with a running total, the two-way corrections, then
    the totals and the pattern's bracket and zone.
    """
    running = row["prior"]
    steps = []
    for c in levels["comparisons"]:
        name = c["name"]
        level = row[f"{name}_level"]
        lvl = c["levels"][level] if level < len(c["levels"]) else {"label": str(level)}
        bits, tf = row[f"{name}_bits"], row[f"{name}_tf"]
        running += bits + tf
        steps.append({"name": name, "level": level, "label": lvl["label"],
                      "m": lvl.get("m"), "u": lvl.get("u"), "bits": bits, "tf": tf,
                      "frequency": row[f"{name}_frequency"], "running": running})
    interactions = []
    for name in levels["interactions"]:
        bits = row[name.replace(" x ", "_x_") + "_bits"]
        running += bits
        interactions.append({"name": name, "bits": bits, "running": running})
    return {"prior": row["prior"], "records": levels["records"],
            "threshold": row["threshold"], "steps": steps,
            "interactions": interactions, "weight": row["match_weight"],
            "probability": row["match_probability"],
            "bracket": [row["bracket_low"], row["bracket_high"]],
            "zone": row["zone"], "emitted": bool(row["emitted"])}


class Ledger:
    """The waterfalls compacted for embedding.

    A level's label, rates and bits are the model's, not the pair's, so they are
    kept once per (comparison, level) and a pair stores the level it hit, its
    term-frequency move and the frequency behind it. The page expands that back
    to the full ledger before drawing, so both viewers draw the same shape.
    """

    def __init__(self, levels):
        self.levels = levels
        self.model = None
        self.comparisons = [{"name": c["name"], "levels": []}
                            for c in levels["comparisons"]]
        self.count = 0

    def add(self, row):
        w = ledger(row, self.levels)
        if self.model is None:
            self.model = {"prior": w["prior"], "records": w["records"],
                          "threshold": w["threshold"],
                          "interactions": self.levels["interactions"]}
        steps = []
        for c, step in enumerate(w["steps"]):
            table = self.comparisons[c]["levels"]
            while len(table) <= step["level"]:
                table.append(None)
            if table[step["level"]] is None:
                table[step["level"]] = {"label": step["label"], "bits": step["bits"],
                                        "m": step["m"], "u": step["u"]}
            entry = [step["level"]]
            if step["tf"]:
                entry += [step["tf"], step["frequency"]]
            steps.append(entry)
        self.count += 1
        return {"s": steps, "i": [t["bits"] for t in w["interactions"]],
                "w": w["weight"], "p": w["probability"], "b": w["bracket"],
                "z": w["zone"]}

    def summary(self):
        if self.model is None:
            return None
        return dict(self.model, comparisons=self.comparisons)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="the parquet file(s) the run read, in order")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--clusters", required=True,
                    help="the csv or parquet cpplink cluster wrote")
    ap.add_argument("--predictions", "--edges", dest="predictions",
                    help="the run's predictions -- one csv or parquet file, or the "
                         "shard directory -- for within-cluster weights")
    ap.add_argument("--truth", help="known pairs, to colour members by true entity")
    ap.add_argument("--out", default="clusters.html")
    ap.add_argument("--limit", type=int, default=400, help="clusters to embed (0 = all)")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=200,
                    help="members embedded per cluster; the rest are counted only")
    ap.add_argument("--max-size", type=int, default=0, help="0 = no ceiling")
    ap.add_argument("--sort", default="size", choices=["size", "id", "random"],
                    help="which clusters to embed when --limit cuts the list")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--id", action="append", default=[],
                    help="always embed this cluster id; repeatable")
    ap.add_argument("--waterfalls", help="the csv or parquet cpplink explain "
                                         "--predictions wrote; embeds each "
                                         "prediction's waterfall")
    ap.add_argument("--model", help="the model the run scored with, for the level "
                                    "labels and rates the waterfall shows")
    args = ap.parse_args()
    if bool(args.waterfalls) != bool(args.model):
        ap.error("--waterfalls and --model go together")

    id_column, columns = schema_columns(args.schema)

    assignment = read_assignment(args.clusters)
    chosen, total_clusters = choose_clusters(assignment, args)
    members = read_members(assignment, chosen)
    del assignment

    sizes = {cluster: len(uids) for cluster, uids in members.items()}
    if args.max_rows:
        members = {c: uids[: args.max_rows] for c, uids in members.items()}
    wanted_ids = {uid for cluster in members.values() for uid in cluster}
    table, rows_by_id, records, groups, touched = read_records(
        args.data, id_column, columns, wanted_ids)
    if table is None:
        raise SystemExit("none of the chosen clusters' ids are in the parquet")
    values = {name: table.column(name).to_pylist() for name in columns}
    by_id = {uid: i for i, uid in enumerate(table.column(id_column).to_pylist())}

    truth = read_truth(args.truth, wanted_ids) if args.truth else None
    predictions, predictions_read = ({}, 0)
    by_cluster = {}
    rejected = []
    if args.predictions:
        predictions, predictions_read = read_predictions(
            args.predictions, wanted_ids,
            {u: rows_by_id[u] for u in wanted_ids if u in rows_by_id})
        # A record is in one cluster, so grouping the predictions once is what
        # keeps the loop below linear in them rather than one pass per cluster.
        # A prediction whose ends clustering kept apart is kept too: it was
        # scored above the write threshold and below the clustering one, and
        # that is the pair a reader most wants explained.
        cluster_of = {uid: cluster
                      for cluster, uids in members.items() for uid in uids}
        for (a, b), weight in predictions.items():
            cluster = cluster_of.get(a)
            other = cluster_of.get(b)
            if cluster is None or other is None or a not in by_id or b not in by_id:
                continue
            if cluster == other:
                by_cluster.setdefault(cluster, []).append((a, b, weight))
            else:
                rejected.append((a, b, weight, cluster, other))

    payload_clusters = []
    for cluster in chosen:
        uids = [u for u in members.get(cluster, []) if u in by_id]
        if len(uids) < args.min_size:
            continue
        rows, groups_seen = [], {}
        for uid in uids:
            at = by_id[uid]
            row = {"id": uid, "v": [cell(values[name][at]) for name in columns]}
            if truth is not None:
                group = truth.get(uid, "\x00" + uid)
                row["t"] = groups_seen.setdefault(group, len(groups_seen))
                # The label is per cluster; the group is what says "same entity"
                # for a pair the clustering split across two of them.
                if uid in truth:
                    row["g"] = truth[uid]
            rows.append(row)
        entry = {"id": cluster, "rows": rows}
        if sizes[cluster] > len(rows):
            entry["more"] = sizes[cluster] - len(rows)
        if args.predictions:
            position = {uid: i for i, uid in enumerate(uids)}
            found = []
            for a, b, weight in by_cluster.get(cluster, []):
                if a in position and b in position:
                    i, j = position[a], position[b]
                    found.append([min(i, j), max(i, j), round(weight, 3)])
            entry["edges"] = sorted(found)
        payload_clusters.append(entry)

    payload_rejected = [{"a": a, "b": b, "w": round(weight, 3), "ca": ca, "cb": cb}
                        for a, b, weight, ca, cb in sorted(rejected)]

    # The waterfalls of the embedded predictions, read from the file the run's
    # explain wrote and compacted against the model's level table.
    ledgers = None
    explained = waterfalls_read = 0
    if args.waterfalls and args.predictions:
        ledgers = Ledger(read_model_levels(args.model))
        found, waterfalls_read = read_waterfalls(args.waterfalls, wanted_ids)
        for entry in payload_clusters:
            uid_at = [r["id"] for r in entry["rows"]]
            for edge in entry.get("edges", []):
                row = found.get((uid_at[edge[0]], uid_at[edge[1]]))
                if row is not None:
                    edge.append(ledgers.add(row))
                    explained += 1
        for pair in payload_rejected:
            row = found.get((pair["a"], pair["b"]))
            if row is not None:
                pair["wf"] = ledgers.add(row)
                explained += 1

    payload = {
        "source": ", ".join(os.path.basename(p) for p in args.data),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "columns": columns,
        "totals": {"clusters": total_clusters, "shown": len(payload_clusters),
                   "records": records},
        "clusters": payload_clusters,
        "rejected": payload_rejected,
        "model": ledgers.summary() if ledgers else None,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "cluster_view_template.html")) as handle:
        page = handle.read()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    with open(args.out, "w") as handle:
        handle.write(page.replace("__CLUSTER_DATA__", blob))

    size = os.path.getsize(args.out) / 1e6
    print(f"{len(payload_clusters):,} clusters of {total_clusters:,} "
          f"-> {args.out} ({size:.1f} MB)")
    print(f"{len(wanted_ids):,} records read from {touched:,} of {groups:,} row "
          f"groups over {records:,} rows")
    if args.predictions:
        print(f"{sum(len(c.get('edges', [])) for c in payload_clusters):,} "
              f"within-cluster predictions carried and {len(payload_rejected):,} "
              f"across clusters, {predictions_read:,} scanned")
    if args.waterfalls:
        print(f"{explained:,} waterfalls embedded, {waterfalls_read:,} scanned")


if __name__ == "__main__":
    sys.exit(main())
