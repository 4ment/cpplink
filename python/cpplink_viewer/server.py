# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Serve the cluster viewer over a DuckDB cache of what a run wrote.

    cpplink-viewer --schema s.json --clusters clusters.csv \\
        --predictions predictions.parquet --waterfalls waterfalls.parquet \\
        --model m.json --truth truth.csv data.parquet

The clusters are queried rather than written into the page, so every cluster
of the run is reachable. The left pane lists clusters and the right one has
two tabs for the cluster picked: the members side by side with every
disagreeing cell highlighted, and every prediction touching the cluster with
the one picked drawn as the ledger the scorer produced.

`--clusters` and `--predictions` each name one file, csv or parquet, picked by
the extension; `--predictions` also takes the shard directory, which names
records by row and so costs the pass that names them.

With `--waterfalls` the page also draws the waterfall of the prediction picked.
The file is what `cpplink explain --predictions <file> --out <file>` writes, one
wide row per prediction; it is loaded into the cache beside the predictions and
a click is one lookup, so the ledger is the scorer's own arithmetic and nothing
is computed or spawned here. The level labels and rates come from `--model`.
Predictions clustering kept apart, above the write threshold and below the
clustering one, are kept and listed under both of their clusters as `rejected`.

`--threshold` keeps only the predictions at or above it. With `--clusters` the
file is taken as what `cpplink cluster --threshold` wrote at that threshold and
read as it stands, which is the fast path; without one the predictions are
clustered here, by the same union-find over the same predictions in the same
order, so the cache holds exactly the partition that command writes, named by
the same representatives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .cache import Options, announce, open_cache, quoted
from .inputs import cell, ledger, read_model_levels

# The list's order: the chosen key in either direction, a cluster with no
# value for it (no predictions, so no weakest edge) last either way, and the
# ties broken the same way whichever direction the key runs.
ORDERS = {
    "discord": ("discord", "size DESC, id"),
    "size": ("size", "discord DESC, id"),
    "weakest": ("weakest", "size DESC, id"),
    "id": ("id", ""),
}

FILTERS = {
    "all": "TRUE",
    "split": "discord > 0",
    "transitive": "size * (size - 1) / 2 > edge_count",
    "mixed": "entities > 1",
}

REJECTED_LIMIT = 1000  # rejected predictions listed per cluster, weakest first
LIST_LIMIT = 500  # clusters a list request may ask for at once


def order_clause(sort, desc):
    key, ties = ORDERS.get(sort, ORDERS["discord"])
    clause = f"{key} {'DESC' if desc else 'ASC'} NULLS LAST"
    return clause + (", " + ties if ties else "")


def like_pattern(text):
    escaped = text.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "%" + escaped + "%"


class Viewer:
    """The queries the page makes, over one shared read-only connection."""

    def __init__(self, conn, columns, source, max_rows, has_truth, levels=None):
        self.conn = conn
        self.columns = columns
        self.source = source
        self.max_rows = max_rows
        self.has_truth = has_truth
        self.levels = levels  # the model's level table, when waterfalls are held
        self.lock = threading.Lock()
        with self.lock:
            self.totals = conn.execute(
                "SELECT count(*), coalesce(sum(size), 0) FROM stats"
            ).fetchone()
            self.pair_total = conn.execute("SELECT count(*) FROM pairs").fetchone()[0]
            self.waterfall_columns = [
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'waterfalls' ORDER BY ordinal_position"
                ).fetchall()
            ]

    @classmethod
    def open(cls, opts, log=announce):
        """A viewer over the cache `opts` names, built first if it has to be."""
        reason = opts.check()
        if reason:
            raise SystemExit(reason)
        conn, _, columns = open_cache(opts, log)
        levels = read_model_levels(opts.model) if opts.model else None
        source = ", ".join(os.path.basename(p) for p in opts.data)
        return cls(conn, columns, source, opts.max_rows, bool(opts.truth), levels)

    def query(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def summary(self):
        return {
            "columns": self.columns,
            "source": self.source,
            "totals": {
                "clusters": self.totals[0],
                "records": self.totals[1],
                "pairs": self.pair_total,
            },
            "has_truth": self.has_truth,
            "has_model": bool(self.waterfall_columns),
        }

    def waterfall(self, a, b):
        """The ledger of one prediction, from the wide row the run's explain wrote."""
        if not self.waterfall_columns or self.levels is None:
            return None
        rows = self.query("SELECT * FROM waterfalls WHERE id_a = ? AND id_b = ?", [a, b])
        if not rows:
            return None
        row = dict(zip(self.waterfall_columns, rows[0], strict=True))
        return ledger(row, self.levels)

    def rejected(self, cluster_id):
        """The predictions clustering overruled that touch one cluster.

        Each names the other cluster its second record went to, and the truth's
        verdict where there is one, so the page can list them beside the
        cluster's own edges without a lookup per row.
        """
        rows = self.query(
            "SELECT p.a, p.b, p.weight, p.cluster_a, p.cluster_b, ta.grp, tb.grp "
            "FROM pairs p LEFT JOIN truth ta ON ta.uid = p.a "
            "LEFT JOIN truth tb ON tb.uid = p.b "
            "WHERE (p.cluster_a = ? OR p.cluster_b = ?) AND p.cluster_a <> p.cluster_b "
            f"ORDER BY p.weight, p.a, p.b LIMIT {REJECTED_LIMIT + 1}",
            [cluster_id, cluster_id],
        )
        more = max(0, len(rows) - REJECTED_LIMIT)
        found = []
        for a, b, w, ca, cb, ga, gb in rows[:REJECTED_LIMIT]:
            same = (ga is not None and ga == gb) if self.has_truth else None
            found.append(
                {
                    "a": a,
                    "b": b,
                    "w": round(w, 3),
                    "other": cb if ca == cluster_id else ca,
                    "same": same,
                }
            )
        return found, more

    def pair(self, a, b):
        head = self.query(
            "SELECT a, b, weight, cluster_a, cluster_b FROM pairs WHERE a = ? AND b = ?",
            [a, b],
        )
        if not head:
            return None
        picked = ", ".join(quoted(c) for c in self.columns)
        rows = {
            r[0]: r
            for r in self.query(
                f"SELECT uid, {picked} FROM records WHERE uid IN (?, ?)", [a, b]
            )
        }
        if a not in rows or b not in rows:
            return None
        answer = {
            "a": a,
            "b": b,
            "w": round(head[0][2], 3),
            "ca": head[0][3],
            "cb": head[0][4],
            "rows": [
                {"id": uid, "v": [cell(v) for v in rows[uid][1:]]} for uid in (a, b)
            ],
        }
        if self.has_truth:
            groups = dict(
                self.query("SELECT uid, grp FROM truth WHERE uid IN (?, ?)", [a, b])
            )
            same = a in groups and b in groups and groups[a] == groups[b]
            answer["rows"][0]["t"] = 0
            answer["rows"][1]["t"] = 0 if same else 1
        answer["waterfall"] = self.waterfall(a, b)
        return answer

    def listing(self, q, field, column, sort, desc, only, offset, limit):
        """One page of the cluster list.

        `field` says what `q` is read against: `any` is a substring of the
        concatenated record, `cluster` of the cluster id, `col` of the named
        column, and `id` a comma-separated list of record ids matched whole, in
        which case the answer also says which of them named a record.
        """
        where = FILTERS.get(only, "TRUE")
        params = []
        found = None
        if q and field == "id":
            ids = list(dict.fromkeys(s.strip() for s in q.split(",") if s.strip()))
            marks = ", ".join("?" * len(ids))
            where += (
                f" AND id IN (SELECT cluster_id FROM records "
                f"WHERE CAST(uid AS VARCHAR) IN ({marks}))"
            )
            params.extend(ids)
            found = [
                r[0]
                for r in self.query(
                    f"SELECT DISTINCT CAST(uid AS VARCHAR) FROM records "
                    f"WHERE CAST(uid AS VARCHAR) IN ({marks})",
                    ids,
                )
            ]
        elif q and field == "cluster":
            where += " AND lower(id) LIKE ? ESCAPE '\\'"
            params.append(like_pattern(q))
        elif q and field == "col":
            if column not in self.columns:
                raise ValueError(f"no column {column!r}")
            where += (
                f" AND id IN (SELECT cluster_id FROM records "
                f"WHERE lower(CAST({quoted(column)} AS VARCHAR)) LIKE ? ESCAPE '\\')"
            )
            params.append(like_pattern(q))
        elif q:
            where += (
                " AND id IN (SELECT cluster_id FROM records "
                "WHERE text LIKE ? ESCAPE '\\')"
            )
            params.append(like_pattern(q))
        matched = self.query(f"SELECT count(*) FROM stats WHERE {where}", params)[0][0]
        rows = self.query(
            f"SELECT id, size, discord, edge_count, weakest, entities FROM stats "
            f"WHERE {where} ORDER BY {order_clause(sort, desc)} "
            f"LIMIT {int(limit)} OFFSET {int(offset)}",
            params,
        )
        answer = {
            "matched": matched,
            "clusters": [
                {
                    "id": r[0],
                    "size": r[1],
                    "discord": r[2],
                    "edge_count": r[3],
                    "weakest": r[4],
                    "entities": r[5],
                }
                for r in rows
            ],
        }
        if found is not None:
            answer["found"] = found
        return answer

    def cluster(self, cluster_id):
        head = self.query(
            "SELECT id, size, discord, edge_count, weakest, entities "
            "FROM stats WHERE id = ?",
            [cluster_id],
        )
        if not head:
            return None
        picked = ", ".join(quoted(c) for c in self.columns)
        rows = self.query(
            f"SELECT uid, {picked} FROM records WHERE cluster_id = ? "
            f"ORDER BY uid LIMIT {int(self.max_rows)}",
            [cluster_id],
        )
        ids = [r[0] for r in rows]
        groups, labels = {}, {}
        if self.has_truth:
            for uid, grp in self.query(
                "SELECT uid, grp FROM truth WHERE uid IN "
                "(SELECT uid FROM records WHERE cluster_id = ?)",
                [cluster_id],
            ):
                groups[uid] = grp
            for uid in ids:
                key = groups.get(uid, "x" + uid)
                labels.setdefault(key, len(labels))
        # The whole cluster's distinct counts, not the page of rows shown.
        counts = self.query(
            "SELECT "
            + ", ".join(f"count(DISTINCT {quoted(c)})" for c in self.columns)
            + " FROM records WHERE cluster_id = ?",
            [cluster_id],
        )[0]
        edges = self.query(
            "SELECT a, b, weight FROM edges WHERE cluster_id = ? AND a IN "
            "(SELECT uid FROM records WHERE cluster_id = ?)",
            [cluster_id, cluster_id],
        )
        keep = set(ids)
        answer = {
            "id": head[0][0],
            "size": head[0][1],
            "discord": head[0][2],
            "edge_count": head[0][3],
            "weakest": head[0][4],
            "entities": head[0][5],
            "dist": list(counts),
            "rows": [{"id": r[0], "v": [cell(v) for v in r[1:]]} for r in rows],
            "edges": [
                [a, b, round(w, 3)] for a, b, w in edges if a in keep and b in keep
            ],
        }
        if self.has_truth:
            for row in answer["rows"]:
                row["t"] = labels[groups.get(row["id"], "x" + row["id"])]
        answer["rejected"], more = self.rejected(cluster_id)
        if more:
            answer["rejected_more"] = more
        return answer


def page_html():
    """The viewer page, shipped beside this module."""
    return (
        resources.files(__package__).joinpath("viewer.html").read_text(encoding="utf-8")
    )


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
                    self.send_json(
                        viewer.listing(
                            query.get("q", ""),
                            query.get("field", "any"),
                            query.get("column", ""),
                            query.get("sort", "discord"),
                            query.get("dir", "desc") == "desc",
                            query.get("only", "all"),
                            int(query.get("offset", 0)),
                            min(int(query.get("limit", 100)), LIST_LIMIT),
                        )
                    )
                elif url.path == "/api/cluster":
                    found = viewer.cluster(query.get("id", ""))
                    self.send_json(
                        found or {"error": "no such cluster"}, 200 if found else 404
                    )
                elif url.path == "/api/pair":
                    found = viewer.pair(query.get("a", ""), query.get("b", ""))
                    self.send_json(
                        found or {"error": "no such prediction"}, 200 if found else 404
                    )
                else:
                    self.send_json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as trouble:  # a bad query should not kill the server
                self.send_json({"error": str(trouble)}, 500)

    return Handler


def make_server(viewer, port=0, host="127.0.0.1"):
    """An HTTP server over `viewer`, bound and not yet serving; port 0 picks a
    free one, which is what a test wants."""
    return ThreadingHTTPServer((host, port), handler_for(viewer, page_html()))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="cpplink-viewer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("data", nargs="+", help="the parquet file(s) the run read, in order")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--clusters", help="the csv or parquet cpplink cluster wrote")
    ap.add_argument(
        "--predictions",
        help="the run's predictions -- one csv or parquet file, or the shard "
        "directory -- for within-cluster weights",
    )
    ap.add_argument("--truth", help="known pairs, to colour members by true entity")
    ap.add_argument(
        "--waterfalls",
        help="the csv or parquet cpplink explain --predictions wrote; draws each "
        "prediction's waterfall",
    )
    ap.add_argument(
        "--model",
        help="the model the run scored with, for the level labels and rates the "
        "waterfall shows",
    )
    ap.add_argument("--cache", default=Options.cache)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cache first")
    ap.add_argument("--min-size", type=int, default=Options.min_size)
    ap.add_argument(
        "--max-size", type=int, default=Options.max_size, help="0 = no ceiling"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        help="keep only the predictions at or above this match_weight; with "
        "--clusters that file is read as the clustering at this threshold, "
        "without one the predictions are clustered here exactly as "
        "cpplink cluster --threshold would",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=Options.max_rows,
        help="members shown per cluster; the rest are counted only",
    )
    ap.add_argument(
        "--memory", default=Options.memory, help="what DuckDB may use while building"
    )
    ap.add_argument(
        "--threads", type=int, default=Options.threads, help="0 = DuckDB's own default"
    )
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="open a browser on it")
    args = ap.parse_args(argv)
    opts = Options(
        data=args.data,
        schema=args.schema,
        clusters=args.clusters,
        predictions=args.predictions,
        truth=args.truth,
        waterfalls=args.waterfalls,
        model=args.model,
        cache=args.cache,
        rebuild=args.rebuild,
        min_size=args.min_size,
        max_size=args.max_size,
        threshold=args.threshold,
        max_rows=args.max_rows,
        memory=args.memory,
        threads=args.threads,
    )
    reason = opts.check()
    if reason:
        ap.error(reason)
    return opts, args.port, args.open


def main(argv=None):
    opts, port, open_browser = parse_args(argv)
    viewer = Viewer.open(opts)
    server = make_server(viewer, port)
    where = f"http://127.0.0.1:{server.server_address[1]}/"
    clusters, records = viewer.totals
    announce(f"{clusters:,} clusters over {records:,} records at {where}")
    if open_browser:
        webbrowser.open(where)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
