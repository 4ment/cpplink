# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Open a csv or parquet file through DuckDB and map its types to cpplink's.

cpplink reads parquet only, and its loader is strict about Arrow types: a
string column must be ``string`` (not ``large_string``, not an integer), a
date must be ``date32``, ``date64`` or a timestamp, a double must be ``double``
or ``float``, a boolean ``bool``, a list ``list<string>``, and the unique id
a string. So a csv, or a parquet holding integer ids or postcodes, has to be
rewritten with the types the user picked before the binary can see it. That
rewrite is ``write_typed_parquet``; everything else here is reading.
"""

import os
import re

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# The types a schema column may declare.
CPPLINK_TYPES = ["string", "string_list", "date", "double", "boolean"]

# A column whose DuckDB type is one of these has no single obvious cpplink
# type, so the mapping proposes one and the page shows the choice as a choice.
INTEGER_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
                 "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT")


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def cpplink_type_for(duckdb_type):
    """The cpplink type a DuckDB column type maps to, and whether that was a
    guess. Integers are strings by default: an integer postcode, phone or id
    is an identifier, not a quantity, and blocking needs discrete agreement."""
    t = duckdb_type.upper()
    if t.endswith("[]"):
        return "string_list", False
    if t in ("VARCHAR", "TEXT", "STRING", "UUID", "BLOB"):
        return "string", False
    if t.startswith("DATE") or t.startswith("TIMESTAMP"):
        return "date", False
    if t in ("DOUBLE", "FLOAT", "REAL") or t.startswith("DECIMAL"):
        return "double", False
    if t == "BOOLEAN":
        return "boolean", False
    if t in INTEGER_TYPES:
        return "string", True
    return "string", True


def arrow_type_accepted(arrow_type, cpplink_type):
    """Whether the loader reads this Arrow type as this cpplink type as-is."""
    if cpplink_type == "string":
        return pa.types.is_string(arrow_type)
    if cpplink_type == "date":
        return (pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type) or
                pa.types.is_timestamp(arrow_type))
    if cpplink_type == "double":
        return pa.types.is_float64(arrow_type) or pa.types.is_float32(arrow_type)
    if cpplink_type == "boolean":
        return pa.types.is_boolean(arrow_type)
    if cpplink_type == "string_list":
        return pa.types.is_list(arrow_type) and pa.types.is_string(arrow_type.value_type)
    return False


class Dataset:
    """One file behind a DuckDB view named ``data``.

    ``columns`` is the ordered list of file column names, ``duckdb_types`` the
    type DuckDB read or sniffed for each, and ``arrow_types`` the Arrow type
    for a parquet file (``None`` for csv, which has no types of its own).
    """

    def __init__(self, path, csv_options=None):
        self.path = path
        self.kind = "parquet" if path.lower().endswith((".parquet", ".pq")) else "csv"
        self.conn = duckdb.connect()
        self.csv_options = dict(csv_options or {})
        if self.kind == "parquet":
            self.conn.execute(
                f"CREATE VIEW data AS SELECT * FROM read_parquet({sql_string(path)})")
            self.arrow_types = {f.name: f.type for f in pq.read_schema(path)}
        else:
            opts = ", ".join(f"{k}={v}" for k, v in self.csv_options.items())
            source = f"read_csv({sql_string(path)}" + (f", {opts}" if opts else "") + ")"
            self.conn.execute(f"CREATE VIEW data AS SELECT * FROM {source}")
            self.arrow_types = {}
        described = self.conn.execute("DESCRIBE data").fetchall()
        self.columns = [row[0] for row in described]
        self.duckdb_types = {row[0]: row[1] for row in described}
        self.rows = self.conn.execute("SELECT count(*) FROM data").fetchone()[0]

    def query(self, sql, params=()):
        return self.conn.execute(sql, params)

    def ensure_sample(self, rows=100_000):
        """A reservoir sample of the file as the table ``sample``, drawn once:
        every per-column probe reads it rather than scanning the file again."""
        if getattr(self, "_sample_rows", None) == rows:
            return
        if self.rows <= rows:
            self.conn.execute("CREATE OR REPLACE TABLE sample AS SELECT * FROM data")
        else:
            self.conn.execute(f"CREATE OR REPLACE TABLE sample AS SELECT * FROM data "
                              f"USING SAMPLE reservoir({int(rows)} ROWS) REPEATABLE (1)")
        self._sample_rows = rows

    def proposed_types(self):
        """{column: (cpplink type, guessed)} for every column of the file."""
        return {c: cpplink_type_for(self.duckdb_types[c]) for c in self.columns}

    def needs_cast(self, column, cpplink_type):
        """Whether the binary could read this column from the file as-is."""
        if self.kind != "parquet":
            return True
        return not arrow_type_accepted(self.arrow_types[column], cpplink_type)

    def head(self, n=20):
        return self.conn.execute(f"SELECT * FROM data LIMIT {int(n)}").df()


def sql_string(text):
    return "'" + text.replace("'", "''") + "'"


def cast_expression(column, cpplink_type, options=None):
    """The SQL that turns a file column into the cpplink type asked for.

    ``options`` may carry ``sentinels`` (values read as missing), ``date_format``
    (a strptime format for a string date) and ``list_separator`` (how a string
    splits into a list). Every cast is a TRY_CAST, so an unparseable value is a
    null rather than a failed export; ``cast_failures`` counts them first.
    """
    options = options or {}
    q = quoted(column)
    # Trim and treat blanks as missing: the record store does the same at load.
    text = f"NULLIF(trim(CAST({q} AS VARCHAR)), '')"
    sentinels = [s for s in options.get("sentinels", []) if s != ""]
    if sentinels:
        listed = ", ".join(sql_string(s.lower()) for s in sentinels)
        text = f"CASE WHEN lower({text}) IN ({listed}) THEN NULL ELSE {text} END"
    if cpplink_type == "string":
        return text
    if cpplink_type == "double":
        return f"TRY_CAST({text} AS DOUBLE)"
    if cpplink_type == "boolean":
        return f"TRY_CAST({text} AS BOOLEAN)"
    if cpplink_type == "date":
        fmt = options.get("date_format")
        if fmt:
            return f"CAST(try_strptime({text}, {sql_string(fmt)}) AS DATE)"
        return f"TRY_CAST({text} AS DATE)"
    if cpplink_type == "string_list":
        sep = options.get("list_separator")
        if sep is None:
            # Already a list in the file: keep it, as strings.
            return f"CAST({q} AS VARCHAR[])"
        return (f"list_filter(list_transform(string_split({text}, {sql_string(sep)}), "
                f"x -> trim(x)), x -> x <> '')")
    raise ValueError(f"unknown cpplink type {cpplink_type!r}")


def cast_failures(ds, column, cpplink_type, options=None):
    """Rows that hold a value but cast to null: the parse failures an export
    would silently turn into missing values."""
    q = quoted(column)
    expr = cast_expression(column, cpplink_type, options)
    sql = (f"SELECT count(*) FROM data WHERE {q} IS NOT NULL "
           f"AND trim(CAST({q} AS VARCHAR)) <> '' AND ({expr}) IS NULL")
    return ds.query(sql).fetchone()[0]


def write_typed_parquet(ds, out_path, types, unique_id, options=None, include=None):
    """Write the file cpplink will read: every included column cast to its
    cpplink type, the id as a string, and a row-number id when there is none.

    ``types`` is {column: cpplink type}; ``options`` is {column: cast options};
    ``include`` restricts which columns are written (the id is always written).
    Returns the id column name in the written file.
    """
    options = options or {}
    include = list(include) if include is not None else list(types)
    parts = []
    if unique_id:
        parts.append(f"CAST({quoted(unique_id)} AS VARCHAR) AS {quoted(unique_id)}")
        id_name = unique_id
    else:
        id_name = "unique_id"
        while id_name in ds.columns:
            id_name += "_"
        parts.append(f"CAST(row_number() OVER () AS VARCHAR) AS {quoted(id_name)}")
    for column in include:
        if column == unique_id:
            continue
        expr = cast_expression(column, types[column], options.get(column))
        parts.append(f"{expr} AS {quoted(column)}")
    sql = (f"COPY (SELECT {', '.join(parts)} FROM data) TO {sql_string(out_path)} "
           f"(FORMAT PARQUET)")
    ds.query(sql)
    return id_name


def sniff_csv(path):
    """DuckDB's reading of a csv's dialect, for the page to show and override."""
    conn = duckdb.connect()
    row = conn.execute(f"SELECT * FROM sniff_csv({sql_string(path)})").fetchone()
    names = [d[0] for d in conn.description]
    return dict(zip(names, row))


def default_output_path(path):
    """Where the typed parquet goes when the user does not say: beside the
    input, with a suffix that says the studio wrote it."""
    base = re.sub(r"\.(csv|tsv|txt|parquet|pq)$", "", path, flags=re.IGNORECASE)
    return base + ".studio.parquet"


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None
