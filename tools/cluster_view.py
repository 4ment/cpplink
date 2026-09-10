#!/usr/bin/env python3
# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Build a self-contained HTML viewer for the clusters a cpplink run produced.

    cpplink predict --schema s.json --model m.json --out predictions.parquet data.parquet
    cpplink cluster --schema s.json --predictions predictions.parquet \
        --out clusters.csv data.parquet
    tools/cluster_view.py --schema s.json --clusters clusters.csv \
        --predictions predictions.parquet --out clusters.html data.parquet

Reads the cluster assignment, pulls each member's column values back out of the
parquet, and writes one HTML file holding the clusters it selected. Open it in a
browser: the left pane lists clusters, the right one shows the members side by
side with every disagreeing cell highlighted, so what a cluster has in common is
the part that is not highlighted.

The `network` checkbox in the header draws the selected cluster's predictions as
a graph, which is where a chain shows itself as a chain. It is off by default and
the choice is remembered, because the layout is the one quadratic thing the page
does and most clusters are read without it.

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
    args = ap.parse_args()

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
    if args.predictions:
        predictions, predictions_read = read_predictions(
            args.predictions, wanted_ids,
            {u: rows_by_id[u] for u in wanted_ids if u in rows_by_id})
        # A record is in one cluster, so grouping the predictions once is what
        # keeps the loop below linear in them rather than one pass per cluster.
        cluster_of = {uid: cluster
                      for cluster, uids in members.items() for uid in uids}
        for (a, b), weight in predictions.items():
            cluster = cluster_of.get(a)
            if cluster is not None and cluster == cluster_of.get(b):
                by_cluster.setdefault(cluster, []).append((a, b, weight))

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

    payload = {
        "source": ", ".join(os.path.basename(p) for p in args.data),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "columns": columns,
        "totals": {"clusters": total_clusters, "shown": len(payload_clusters)},
        "clusters": payload_clusters,
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
        print(f"{len(predictions):,} within-cluster predictions carried, "
              f"{predictions_read:,} scanned")


if __name__ == "__main__":
    sys.exit(main())
