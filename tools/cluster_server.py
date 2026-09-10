#!/usr/bin/env python3
# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Serve the cluster viewer over a DuckDB cache, for runs too large to embed.

    tools/cluster_server.py --schema s.json --clusters clusters.csv \
        --edges edges --truth truth.csv data.parquet

Same page as cluster_view.py, but the clusters are queried rather than written
into the file, so every cluster of the run is reachable instead of a sample.

The cache is built once and reused until an input changes. It holds only what
the viewer can ever show -- the clustered records, never the singletons -- so
it is a fraction of the dataset: at 20M records with 2.9M of them clustered,
the parquet is read exactly once and every later request touches a table seven
times smaller than the file.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_view import (EDGE_DTYPE, EDGE_MAGIC, cell,  # noqa: E402
                          read_truth, schema_columns)

EDGE_CHUNK = 1 << 20
CACHE_VERSION = 2


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def fingerprint(args, columns):
    """What the cache was built from, so a changed input rebuilds it."""
    parts = [CACHE_VERSION, columns, args.max_rows]
    for path in [args.clusters, args.truth] + list(args.data):
        if path:
            parts.append([os.path.abspath(path), os.path.getmtime(path),
                          os.path.getsize(path)])
    if args.edges:
        for name in sorted(os.listdir(args.edges)):
            if name.endswith(".bin"):
                full = os.path.join(args.edges, name)
                parts.append([full, os.path.getmtime(full), os.path.getsize(full)])
    return json.dumps(parts, sort_keys=True)


def load_edges(conn, directory):
    """Every shard edge into `raw_edges`, a chunk at a time.

    The shards hold row indices rather than ids, which is why this is worth
    doing once: the join that names them costs a pass over the record ids, and
    a served request must not pay it.
    """
    conn.execute("CREATE TABLE raw_edges (a UINTEGER, b UINTEGER, w DOUBLE)")
    total = 0
    for name in sorted(n for n in os.listdir(directory) if n.endswith(".bin")):
        with open(os.path.join(directory, name), "rb") as handle:
            if handle.read(len(EDGE_MAGIC)) != EDGE_MAGIC:
                raise SystemExit(f"{name}: not a cpplink edge shard")
            while True:
                data = handle.read(EDGE_CHUNK * EDGE_DTYPE.itemsize)
                if not data:
                    break
                block = np.frombuffer(data, dtype=EDGE_DTYPE)
                chunk = pa.table({"a": block["a"], "b": block["b"], "w": block["w"]})
                conn.register("chunk", chunk)
                conn.execute("INSERT INTO raw_edges SELECT a, b, w FROM chunk")
                conn.unregister("chunk")
                total += block.size
    return total


def build(conn, args, id_column, columns, say):
    """Fill the cache: members, their records, their edges, per-cluster stats."""
    files = [os.path.abspath(p) for p in args.data]
    picked = ", ".join(quoted(c) for c in columns)

    say("reading the cluster assignment")
    conn.execute(f"""
        CREATE TABLE members AS
        SELECT unique_id AS uid, cluster_id, cluster_size AS size
        FROM read_csv('{args.clusters}', header = true,
                      columns = {{'unique_id': 'VARCHAR', 'cluster_id': 'VARCHAR',
                                  'cluster_size': 'BIGINT'}})
        WHERE cluster_size >= {args.min_size}
    """)

    say("reading the clustered records")
    conn.execute(f"""
        CREATE TABLE records AS
        SELECT m.cluster_id, d.{quoted(id_column)} AS uid, {picked},
               lower(concat_ws(' ', d.{quoted(id_column)}, {picked})) AS text
        FROM read_parquet({files}) d
        JOIN members m ON m.uid = d.{quoted(id_column)}
    """)

    if args.edges:
        say("naming the edges")
        # A row index in a shard is a position in the inputs read in order, and
        # file_row_number is that position inside one file, so the offsets that
        # separate the inputs are the only thing to carry across.
        conn.execute("CREATE TABLE rowmap (rid BIGINT, uid VARCHAR)")
        offset = 0
        for path in files:
            conn.execute(f"""
                INSERT INTO rowmap
                SELECT {offset} + file_row_number, {quoted(id_column)}
                FROM read_parquet('{path}', file_row_number = true)
            """)
            offset += pq.ParquetFile(path).metadata.num_rows
        load_edges(conn, args.edges)
        conn.execute("""
            CREATE TABLE edges AS
            SELECT m.cluster_id, ra.uid AS a, rb.uid AS b, e.w AS weight
            FROM raw_edges e
            JOIN rowmap ra ON ra.rid = e.a
            JOIN rowmap rb ON rb.rid = e.b
            JOIN members m ON m.uid = ra.uid
        """)
        conn.execute("DROP TABLE raw_edges")
        conn.execute("DROP TABLE rowmap")
        conn.execute("CREATE INDEX edges_by_cluster ON edges (cluster_id)")
    else:
        conn.execute("CREATE TABLE edges (cluster_id VARCHAR, a VARCHAR, "
                     "b VARCHAR, weight DOUBLE)")

    if args.truth:
        say("closing the known pairs")
        uids = [row[0] for row in conn.execute("SELECT uid FROM members").fetchall()]
        groups = read_truth(args.truth, set(uids))
        table = pa.table({"uid": pa.array(list(groups.keys()), pa.string()),
                          "grp": pa.array(list(groups.values()), pa.int64())})
        conn.register("groups", table)
        conn.execute("CREATE TABLE truth AS SELECT uid, grp FROM groups")
        conn.unregister("groups")
    else:
        conn.execute("CREATE TABLE truth (uid VARCHAR, grp BIGINT)")

    say("counting what each cluster agrees on")
    # count(DISTINCT x) skips nulls, which is the page's rule: a column with one
    # value and some gaps still agrees.
    splits = " + ".join(
        f"CASE WHEN count(DISTINCT {quoted(c)}) > 1 THEN 1 ELSE 0 END" for c in columns)
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
                   count(DISTINCT coalesce(CAST(t.grp AS VARCHAR), 'x' || r.uid)) AS entities
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


ORDERS = {
    "discord": "discord DESC, size DESC, id",
    "size": "size DESC, discord DESC, id",
    "weakest": "weakest NULLS LAST, size DESC, id",
    "id": "id",
}

FILTERS = {
    "all": "TRUE",
    "split": "discord > 0",
    "transitive": "size * (size - 1) / 2 > edge_count",
    "mixed": "entities > 1",
}


class Viewer:
    """The queries the page makes, over one shared read-only connection."""

    def __init__(self, conn, columns, source, max_rows, has_truth):
        self.conn = conn
        self.columns = columns
        self.source = source
        self.max_rows = max_rows
        self.has_truth = has_truth
        self.lock = threading.Lock()
        with self.lock:
            self.totals = conn.execute(
                "SELECT count(*), coalesce(sum(size), 0) FROM stats").fetchone()

    def query(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def summary(self):
        return {"columns": self.columns, "source": self.source,
                "totals": {"clusters": self.totals[0], "records": self.totals[1]},
                "has_truth": self.has_truth}

    def listing(self, q, sort, only, offset, limit):
        where = FILTERS.get(only, "TRUE")
        params = []
        if q:
            where += (" AND id IN (SELECT cluster_id FROM records "
                      "WHERE text LIKE ? ESCAPE '\\')")
            params.append("%" + q.lower().replace("\\", "\\\\")
                          .replace("%", "\\%").replace("_", "\\_") + "%")
        matched = self.query(f"SELECT count(*) FROM stats WHERE {where}", params)[0][0]
        rows = self.query(
            f"SELECT id, size, discord, edge_count, weakest, entities FROM stats "
            f"WHERE {where} ORDER BY {ORDERS.get(sort, ORDERS['discord'])} "
            f"LIMIT {int(limit)} OFFSET {int(offset)}", params)
        return {"matched": matched,
                "clusters": [{"id": r[0], "size": r[1], "discord": r[2],
                              "edge_count": r[3], "weakest": r[4], "entities": r[5]}
                             for r in rows]}

    def cluster(self, cluster_id):
        head = self.query("SELECT id, size, discord, edge_count, weakest, entities "
                          "FROM stats WHERE id = ?", [cluster_id])
        if not head:
            return None
        picked = ", ".join(quoted(c) for c in self.columns)
        rows = self.query(
            f"SELECT uid, {picked} FROM records WHERE cluster_id = ? "
            f"ORDER BY uid LIMIT {int(self.max_rows)}", [cluster_id])
        ids = [r[0] for r in rows]
        groups, labels = {}, {}
        if self.has_truth:
            for uid, grp in self.query(
                    "SELECT uid, grp FROM truth WHERE uid IN "
                    "(SELECT uid FROM records WHERE cluster_id = ?)", [cluster_id]):
                groups[uid] = grp
            for uid in ids:
                key = groups.get(uid, "x" + uid)
                labels.setdefault(key, len(labels))
        # The whole cluster's distinct counts, not the page of rows shown.
        counts = self.query(
            "SELECT " + ", ".join(f"count(DISTINCT {quoted(c)})" for c in self.columns)
            + " FROM records WHERE cluster_id = ?", [cluster_id])[0]
        edges = self.query(
            "SELECT a, b, weight FROM edges WHERE cluster_id = ? AND a IN "
            "(SELECT uid FROM records WHERE cluster_id = ?)", [cluster_id, cluster_id])
        keep = set(ids)
        answer = {
            "id": head[0][0], "size": head[0][1], "discord": head[0][2],
            "edge_count": head[0][3], "weakest": head[0][4], "entities": head[0][5],
            "dist": list(counts),
            "rows": [{"id": r[0], "v": [cell(v) for v in r[1:]]} for r in rows],
            "edges": [[a, b, round(w, 3)] for a, b, w in edges
                      if a in keep and b in keep],
        }
        if self.has_truth:
            for row in answer["rows"]:
                row["t"] = labels[groups.get(row["id"], "x" + row["id"])]
        return answer


def handler_for(viewer, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    body = page.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/summary":
                    self.send_json(viewer.summary())
                elif url.path == "/api/list":
                    self.send_json(viewer.listing(
                        query.get("q", ""), query.get("sort", "discord"),
                        query.get("only", "all"), int(query.get("offset", 0)),
                        min(int(query.get("limit", 100)), 500)))
                elif url.path == "/api/cluster":
                    found = viewer.cluster(query.get("id", ""))
                    self.send_json(found or {"error": "no such cluster"},
                                   200 if found else 404)
                else:
                    self.send_json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as trouble:  # a bad query should not kill the server
                self.send_json({"error": str(trouble)}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="the parquet file(s) the run read, in order")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--clusters", required=True, help="clusters.csv from cpplink cluster")
    ap.add_argument("--edges", help="the shard directory, for within-cluster weights")
    ap.add_argument("--truth", help="known pairs, to colour members by true entity")
    ap.add_argument("--cache", default="cluster_view.duckdb")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cache first")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=200,
                    help="members shown per cluster; the rest are counted only")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="open a browser on it")
    args = ap.parse_args()

    id_column, columns = schema_columns(args.schema)
    stamp = fingerprint(args, columns)

    fresh = args.rebuild or not os.path.exists(args.cache)
    if not fresh:
        try:
            conn = duckdb.connect(args.cache, read_only=True)
            fresh = conn.execute("SELECT stamp FROM meta").fetchone()[0] != stamp
            conn.close()
        except Exception:
            fresh = True

    if fresh:
        if os.path.exists(args.cache):
            os.remove(args.cache)
        start = time.time()
        conn = duckdb.connect(args.cache)

        def say(what):
            print(f"  {time.time() - start:6.1f}s  {what}", flush=True)

        print(f"building {args.cache}")
        build(conn, args, id_column, columns, say)
        conn.execute("CREATE TABLE meta (stamp VARCHAR)")
        conn.execute("INSERT INTO meta VALUES (?)", [stamp])
        conn.close()
        size = os.path.getsize(args.cache) / 1e6
        print(f"  {time.time() - start:6.1f}s  done, {size:.0f} MB")

    conn = duckdb.connect(args.cache, read_only=True)
    viewer = Viewer(conn, columns, ", ".join(os.path.basename(p) for p in args.data),
                    args.max_rows, bool(args.truth))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "cluster_view_template.html")) as handle:
        page = handle.read().replace("__CLUSTER_DATA__", "null")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(viewer, page))
    where = f"http://127.0.0.1:{args.port}/"
    print(f"{viewer.totals[0]:,} clusters over {viewer.totals[1]:,} records at {where}")
    if args.open:
        webbrowser.open(where)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    sys.exit(main())
