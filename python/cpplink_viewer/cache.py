# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The DuckDB cache the viewer serves from, built once per set of inputs.

It holds only what the page can ever show: the clustered records, never the
singletons, so it is a fraction of the dataset. At 20M records with 2.9M of them
clustered the parquet is read exactly once and every later request touches a
table seven times smaller than the file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .inputs import EDGE_DTYPE, EDGE_MAGIC, read_truth, schema_columns

EDGE_CHUNK = 1 << 20
CACHE_VERSION = 8


@dataclass
class Options:
    """What the cache is built from and how; the command line fills one."""

    data: list[str]
    schema: str
    clusters: str | None = None
    predictions: str | None = None
    truth: str | None = None
    waterfalls: str | None = None
    model: str | None = None
    cache: str = "cpplink_viewer.duckdb"
    rebuild: bool = False
    min_size: int = 2
    max_size: int = 0
    threshold: float | None = None
    max_rows: int = 200
    memory: str = "4GB"
    threads: int = 0

    def check(self):
        """The reason these options cannot build a cache, or None."""
        if bool(self.waterfalls) != bool(self.model):
            return "--waterfalls and --model go together"
        if self.waterfalls and not self.predictions:
            return "--waterfalls needs --predictions"
        if self.threshold is not None:
            if not self.predictions:
                return "--threshold prunes the predictions, so it needs --predictions"
            if not self.clusters and self.min_size < 2:
                return (
                    "--threshold lists only what a prediction reaches, so a "
                    "singleton is never shown; --min-size must be at least 2"
                )
        elif not self.clusters:
            return "--clusters is needed without --threshold"
        return None


def connect(path, read_only=False):
    """A DuckDB connection, naming the extra to install when there is no DuckDB."""
    try:
        import duckdb
    except ImportError as missing:
        raise SystemExit(
            "the viewer needs duckdb: pip install duckdb, or pip install cpplink[viewer]"
        ) from missing
    return duckdb.connect(path, read_only=read_only)


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def scan(path, types):
    """The duckdb table function that reads one csv or parquet file.

    Both of a run's outputs are one file whose extension picks the format, so
    that is what picks the reader here. The ids are pinned to VARCHAR rather than
    sniffed, because a file of numeric ids would otherwise read as integers and
    join against nothing.
    """
    path = os.path.abspath(path)
    if path.endswith(".parquet"):
        return f"read_parquet('{path}')"
    pinned = ", ".join(f"'{name}': 'VARCHAR'" for name in types)
    return f"read_csv('{path}', header = true, types = {{{pinned}}})"


def fingerprint(opts, columns):
    """What the cache was built from, so a changed input rebuilds it."""
    parts = [
        CACHE_VERSION,
        columns,
        opts.max_rows,
        opts.min_size,
        opts.max_size,
        opts.threshold,
    ]
    for path in [opts.clusters, opts.truth, opts.waterfalls] + list(opts.data):
        if path:
            parts.append(
                [os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path)]
            )
    if opts.predictions:
        if os.path.isdir(opts.predictions):
            for name in sorted(os.listdir(opts.predictions)):
                if name.endswith(".bin"):
                    full = os.path.join(opts.predictions, name)
                    parts.append([full, os.path.getmtime(full), os.path.getsize(full)])
        else:
            parts.append(
                [
                    os.path.abspath(opts.predictions),
                    os.path.getmtime(opts.predictions),
                    os.path.getsize(opts.predictions),
                ]
            )
    return json.dumps(parts, sort_keys=True)


def shard_blocks(directory):
    """Each shard of a directory as one array of its records, sorted by name,
    which is the order `cpplink cluster` reads them in."""
    for name in sorted(n for n in os.listdir(directory) if n.endswith(".bin")):
        with open(os.path.join(directory, name), "rb") as handle:
            if handle.read(len(EDGE_MAGIC)) != EDGE_MAGIC:
                raise SystemExit(f"{name}: not a cpplink prediction shard")
            while True:
                data = handle.read(EDGE_CHUNK * EDGE_DTYPE.itemsize)
                if not data:
                    break
                yield np.frombuffer(data, dtype=EDGE_DTYPE)


def load_shard_edges(conn, directory):
    """Every shard edge into `raw_edges`, a chunk at a time.

    The shards hold row indices rather than ids, which is why this is worth
    doing once: the join that names them costs a pass over the record ids, and
    a served request must not pay it.
    """
    conn.execute("CREATE TABLE raw_edges (a UINTEGER, b UINTEGER, w DOUBLE)")
    total = 0
    for block in shard_blocks(directory):
        chunk = pa.table({"a": block["a"], "b": block["b"], "w": block["w"]})
        conn.register("chunk", chunk)
        conn.execute("INSERT INTO raw_edges SELECT a, b, w FROM chunk")
        conn.unregister("chunk")
        total += block.size
    return total


def read_shard_edges(directory, floor):
    """Every shard edge at or above `floor` as one (a, b, w) table of rows, in
    the order `cpplink cluster` reads them: shard by shard, sorted by name."""
    blocks = [block[block["w"] >= floor] for block in shard_blocks(directory)]
    block = np.concatenate(blocks) if blocks else np.empty(0, dtype=EDGE_DTYPE)
    return pa.table({"a": block["a"], "b": block["b"], "w": block["w"]})


def read_file_edges(path, floor):
    """Every prediction of a merged file at or above `floor` as one (a, b, w)
    table of ids, in file order, which DuckDB would not promise once the build
    connection stops preserving it."""
    if path.endswith(".parquet"):
        table = pq.read_table(path, columns=["id_a", "id_b", "match_weight"])
    else:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                include_columns=["id_a", "id_b", "match_weight"],
                column_types={
                    "id_a": pa.string(),
                    "id_b": pa.string(),
                    "match_weight": pa.float64(),
                },
            ),
        )
    table = table.filter(pc.greater_equal(table["match_weight"], floor))
    return pa.table(
        {
            "a": table["id_a"].cast(pa.string()),
            "b": table["id_b"].cast(pa.string()),
            "w": table["match_weight"],
        }
    )


def id_rows(conn, files, id_column, keys, by):
    """`(uid, rid)` of the records `keys` names, by uid or by rid.

    One pass over the id column, which is the index `cpplink cluster` builds to
    read a merged file back, and what a shard needs to name its rows.
    """
    conn.register("keys", pa.table({"key": keys}))
    parts, offset = [], 0
    for path in files:
        parts.append(f"""
            SELECT CAST({quoted(id_column)} AS VARCHAR) AS uid,
                   {offset} + file_row_number AS rid
            FROM read_parquet('{path}', file_row_number = true)""")
        offset += pq.ParquetFile(path).metadata.num_rows
    found = conn.execute(f"""
        SELECT uid, rid FROM ({" UNION ALL ".join(parts)})
        WHERE {by} IN (SELECT key FROM keys)
    """).to_arrow_table()
    conn.unregister("keys")
    return found


def union_find(edges, count):
    """The core's union-find, edge for edge: union by rank, a tie keeping the
    first end's root, so each cluster comes out named by the representative
    `cpplink cluster` names it by. Returns the root of every index.

    This is the one piece of the core the viewer reimplements, and a test holds
    it to the partition `cpplink cluster --threshold` writes.
    """
    parent = list(range(count))
    rank = [0] * count

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    return [find(x) for x in range(count)]


def recluster(conn, opts, files, id_column, say):
    """`members` and `named_edges` from the predictions at or above the
    threshold, as `cpplink cluster --threshold` would write them.

    A prediction naming a record no input holds is skipped as the core skips
    it, which is why the id pass comes before the union-find rather than after.
    """
    say("reading the predictions at the threshold")
    if os.path.isdir(opts.predictions):
        edges = read_shard_edges(opts.predictions, opts.threshold)
        keys = pc.unique(
            pa.concat_arrays([edges["a"].combine_chunks(), edges["b"].combine_chunks()])
        )
        names = id_rows(conn, files, id_column, keys, "rid")
        by_rid = dict(
            zip(names["rid"].to_pylist(), names["uid"].to_pylist(), strict=True)
        )
        a = [by_rid.get(r) for r in edges["a"].to_pylist()]
        b = [by_rid.get(r) for r in edges["b"].to_pylist()]
    else:
        edges = read_file_edges(opts.predictions, opts.threshold)
        keys = pc.unique(
            pa.concat_arrays([edges["a"].combine_chunks(), edges["b"].combine_chunks()])
        )
        names = id_rows(conn, files, id_column, keys, "uid")
        known = set(names["uid"].to_pylist())
        a = [u if u in known else None for u in edges["a"].to_pylist()]
        b = [u if u in known else None for u in edges["b"].to_pylist()]
    weights = edges["w"].to_pylist()
    kept = [
        (x, y, w)
        for x, y, w in zip(a, b, weights, strict=True)
        if x is not None and y is not None
    ]

    say("clustering them")
    index = {}
    for x, y, _ in kept:
        index.setdefault(x, len(index))
        index.setdefault(y, len(index))
    root = union_find(((index[x], index[y]) for x, y, _ in kept), len(index))
    uids = list(index)
    size = {}
    for r in root:
        size[r] = size.get(r, 0) + 1
    rows = [
        (uids[i], uids[root[i]], size[root[i]])
        for i in range(len(uids))
        if size[root[i]] >= opts.min_size
        and (not opts.max_size or size[root[i]] <= opts.max_size)
    ]
    conn.register(
        "assignment",
        pa.table(
            {
                "uid": pa.array([r[0] for r in rows], pa.string()),
                "cluster_id": pa.array([r[1] for r in rows], pa.string()),
                "size": pa.array([r[2] for r in rows], pa.int64()),
            }
        ),
    )
    conn.execute("CREATE TABLE members AS SELECT uid, cluster_id, size FROM assignment")
    conn.unregister("assignment")
    conn.register(
        "kept",
        pa.table(
            {
                "a": pa.array([k[0] for k in kept], pa.string()),
                "b": pa.array([k[1] for k in kept], pa.string()),
                "weight": pa.array([k[2] for k in kept], pa.float64()),
            }
        ),
    )
    conn.execute("CREATE TABLE named_edges AS SELECT a, b, weight FROM kept")
    conn.unregister("kept")


def build(conn, opts, id_column, columns, say):
    """Fill the cache: members, their records, their edges, per-cluster stats."""
    files = [os.path.abspath(p) for p in opts.data]
    picked = ", ".join(quoted(c) for c in columns)

    if opts.threshold is not None and not opts.clusters:
        recluster(conn, opts, files, id_column, say)
    else:
        say("reading the cluster assignment")
        ceiling = f" AND cluster_size <= {opts.max_size}" if opts.max_size else ""
        conn.execute(f"""
            CREATE TABLE members AS
            SELECT CAST(unique_id AS VARCHAR) AS uid,
                   CAST(cluster_id AS VARCHAR) AS cluster_id,
                   CAST(cluster_size AS BIGINT) AS size
            FROM {scan(opts.clusters, ("unique_id", "cluster_id"))}
            WHERE cluster_size >= {opts.min_size}{ceiling}
        """)

    say("reading the clustered records")
    # A record keeps its store row (`rid`): the position in the inputs read in
    # order, which is what a shard names. file_row_number is that position
    # inside one file, and the offsets between the inputs are the only thing to
    # carry.
    parts, offset = [], 0
    for path in files:
        parts.append(f"""
            SELECT m.cluster_id, d.{quoted(id_column)} AS uid, {picked},
                   lower(concat_ws(' ', d.{quoted(id_column)}, {picked})) AS text,
                   {offset} + file_row_number AS rid
            FROM read_parquet('{path}', file_row_number = true) d
            JOIN members m ON m.uid = d.{quoted(id_column)}""")
        offset += pq.ParquetFile(path).metadata.num_rows
    conn.execute("CREATE TABLE records AS " + " UNION ALL ".join(parts))

    if opts.predictions:
        if opts.threshold is not None and not opts.clusters:
            pass  # `named_edges` is what `recluster` clustered
        elif os.path.isdir(opts.predictions):
            say("naming the predictions")
            # A shard names rows, and the clustered records carry theirs.
            load_shard_edges(conn, opts.predictions)
            conn.execute("""
                CREATE TABLE named_edges AS
                SELECT ra.uid AS a, rb.uid AS b, e.w AS weight
                FROM raw_edges e
                JOIN records ra ON ra.rid = e.a
                JOIN records rb ON rb.rid = e.b
            """)
            conn.execute("DROP TABLE raw_edges")
        else:
            # A merged file already names records by unique_id.
            say("reading the predictions")
            conn.execute(f"""
                CREATE TABLE named_edges AS
                SELECT CAST(id_a AS VARCHAR) AS a, CAST(id_b AS VARCHAR) AS b,
                       match_weight AS weight
                FROM {scan(opts.predictions, ("id_a", "id_b"))}
            """)
        # Both ends clustered, because clustering at a threshold above the one
        # the run wrote at leaves predictions that cross two clusters or land
        # outside every one. A pair across two clusters is kept with both
        # cluster ids, since it is the prediction clustering overruled and the
        # one most worth explaining; a pair with an end no cluster holds has no
        # record to show and is dropped. `--threshold` drops the predictions
        # below it here, so the cache never holds them; when the clusters were
        # computed here they are the components of these very predictions, so
        # every pair lands inside one cluster and none is rejected.
        floor = (
            f" WHERE e.weight >= {opts.threshold}" if opts.threshold is not None else ""
        )
        conn.execute(f"""
            CREATE TABLE pairs AS
            SELECT e.a, e.b, e.weight, ra.cluster_id AS cluster_a,
                   rb.cluster_id AS cluster_b
            FROM named_edges e
            JOIN records ra ON ra.uid = e.a
            JOIN records rb ON rb.uid = e.b{floor}
        """)
        conn.execute("DROP TABLE named_edges")
        conn.execute("""
            CREATE TABLE edges AS
            SELECT cluster_a AS cluster_id, a, b, weight FROM pairs
            WHERE cluster_a = cluster_b
        """)
        conn.execute("CREATE INDEX edges_by_cluster ON edges (cluster_id)")
        # A rejected pair is asked for from either of its clusters.
        conn.execute("CREATE INDEX pairs_by_cluster_a ON pairs (cluster_a)")
        conn.execute("CREATE INDEX pairs_by_cluster_b ON pairs (cluster_b)")
        conn.execute("CREATE INDEX pairs_by_a ON pairs (a)")
        conn.execute("CREATE INDEX pairs_by_b ON pairs (b)")
        if opts.waterfalls:
            # Only the rows the page can ever ask for: the pairs kept above.
            say("reading the waterfalls")
            conn.execute(f"""
                CREATE TABLE waterfalls AS
                SELECT w.* REPLACE (CAST(w.id_a AS VARCHAR) AS id_a,
                                    CAST(w.id_b AS VARCHAR) AS id_b)
                FROM {scan(opts.waterfalls, ("id_a", "id_b"))} w
                JOIN pairs p ON p.a = CAST(w.id_a AS VARCHAR)
                            AND p.b = CAST(w.id_b AS VARCHAR)
            """)
            conn.execute("CREATE INDEX waterfalls_by_pair ON waterfalls (id_a, id_b)")
    else:
        conn.execute(
            "CREATE TABLE edges (cluster_id VARCHAR, a VARCHAR, b VARCHAR, weight DOUBLE)"
        )
        conn.execute(
            "CREATE TABLE pairs (a VARCHAR, b VARCHAR, weight DOUBLE, "
            "cluster_a VARCHAR, cluster_b VARCHAR)"
        )

    if opts.truth:
        say("closing the known pairs")
        uids = [row[0] for row in conn.execute("SELECT uid FROM members").fetchall()]
        groups = read_truth(opts.truth, set(uids))
        table = pa.table(
            {
                "uid": pa.array(list(groups.keys()), pa.string()),
                "grp": pa.array(list(groups.values()), pa.int64()),
            }
        )
        conn.register("groups", table)
        conn.execute("CREATE TABLE truth AS SELECT uid, grp FROM groups")
        conn.unregister("groups")
    else:
        conn.execute("CREATE TABLE truth (uid VARCHAR, grp BIGINT)")

    say("counting what each cluster agrees on")
    # count(DISTINCT x) skips nulls, which is the page's rule: a column with one
    # value and some gaps still agrees.
    splits = " + ".join(
        f"CASE WHEN count(DISTINCT {quoted(c)}) > 1 THEN 1 ELSE 0 END" for c in columns
    )
    conn.execute(f"""
        CREATE TABLE stats AS
        WITH per_cluster AS (
            SELECT cluster_id, count(*) AS size, {splits} AS discord
            FROM records GROUP BY cluster_id),
        per_edge AS (
            SELECT cluster_id, count(*) AS edge_count, min(weight) AS weakest
            FROM edges GROUP BY cluster_id),
        per_truth AS (
            SELECT r.cluster_id,
                   count(DISTINCT coalesce(CAST(t.grp AS VARCHAR), 'x' || r.uid))
                       AS entities
            FROM records r LEFT JOIN truth t ON t.uid = r.uid
            GROUP BY r.cluster_id)
        SELECT c.cluster_id AS id, c.size, c.discord,
               coalesce(e.edge_count, 0) AS edge_count, e.weakest,
               coalesce(t.entities, 0) AS entities
        FROM per_cluster c
        LEFT JOIN per_edge e ON e.cluster_id = c.cluster_id
        LEFT JOIN per_truth t ON t.cluster_id = c.cluster_id
    """)
    conn.execute("CREATE INDEX stats_by_id ON stats (id)")
    conn.execute("CREATE INDEX records_by_cluster ON records (cluster_id)")
    conn.execute("CREATE INDEX records_by_uid ON records (uid)")


def announce(line):
    """The default progress logger: one line, flushed, so a server started in
    the background still shows what it is doing."""
    print(line, flush=True)


def open_cache(opts, log=announce):
    """A read-only connection to the cache for `opts`, built first if it is
    missing, stale, or `opts.rebuild` says so. Returns `(conn, id_column, columns)`."""
    id_column, columns = schema_columns(opts.schema)
    stamp = fingerprint(opts, columns)

    fresh = opts.rebuild or not os.path.exists(opts.cache)
    if not fresh:
        try:
            conn = connect(opts.cache, read_only=True)
            fresh = conn.execute("SELECT stamp FROM meta").fetchone()[0] != stamp
            conn.close()
        except Exception:
            fresh = True

    if fresh:
        if os.path.exists(opts.cache):
            os.remove(opts.cache)
        start = time.time()
        conn = connect(opts.cache)
        conn.execute(f"SET memory_limit = '{opts.memory}'")
        conn.execute("SET preserve_insertion_order = false")
        if opts.threads:
            conn.execute(f"SET threads = {opts.threads}")

        def say(what):
            log(f"  {time.time() - start:6.1f}s  {what}")

        log(f"building {opts.cache}")
        build(conn, opts, id_column, columns, say)
        conn.execute("CREATE TABLE meta (stamp VARCHAR)")
        conn.execute("INSERT INTO meta VALUES (?)", [stamp])
        conn.close()
        size = os.path.getsize(opts.cache) / 1e6
        log(f"  {time.time() - start:6.1f}s  done, {size:.0f} MB")

    conn = connect(opts.cache, read_only=True)
    conn.execute(f"SET memory_limit = '{opts.memory}'")
    return conn, id_column, columns
