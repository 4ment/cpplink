# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The record behind a hit, read out of the parquet by row group.

A hit names a row, a weight and an id, and nothing else: the store is a
re-encoding of the file and does not hand values back. So the page reads the
few rows it is about to draw straight out of the parquet.

What makes that cheap is that the row a hit names *is* the file's row index --
batches are appended in order, so for a single input store row i is parquet row
i -- and parquet is row-group addressable. Ten hits touch at most ten row groups
and each read is a few columns of one group. The alternative, a frame resident
beside the store, doubles the memory of the thing the whole design is careful
about, to display ten rows.

That row-order assumption is load-bearing and silent if it breaks: the page
would draw one person's details beside another person's score. A test in the
core's suite pins it.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence

import pyarrow.parquet as pq


def _display(value: object) -> str:
    """One cell as the page shows it: a list column joined, a null blank."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


class RowFetcher:
    """Rows of one parquet file, by row index, a handful at a time."""

    def __init__(self, path: str, columns: Sequence[str]) -> None:
        self._file = pq.ParquetFile(path)
        self._columns = list(columns)
        # The first row of every row group, so a row index is one bisect.
        self._starts: list[int] = []
        total = 0
        metadata = self._file.metadata
        for group in range(metadata.num_row_groups):
            self._starts.append(total)
            total += metadata.row_group(group).num_rows
        self.num_rows = total

    def group_of(self, row: int) -> int:
        """The row group holding a row, or -1 where the file is shorter."""
        if row < 0 or row >= self.num_rows:
            return -1
        return bisect.bisect_right(self._starts, row) - 1

    def fetch(self, rows: Iterable[int]) -> dict[int, dict[str, str]]:
        """The named rows, keyed by row index.

        Hits are grouped by row group first, so k hits inside one group cost one
        read rather than k. A row past the end of the file is left out rather
        than raising: it means the store and this file have drifted apart, and
        the page says so once instead of failing per row.
        """
        wanted: dict[int, list[int]] = {}
        for row in rows:
            group = self.group_of(row)
            if group >= 0:
                wanted.setdefault(group, []).append(row)

        out: dict[int, dict[str, str]] = {}
        for group, group_rows in wanted.items():
            table = self._file.read_row_group(group, columns=self._columns)
            start = self._starts[group]
            arrays = {name: table.column(name) for name in self._columns}
            for row in group_rows:
                local = row - start
                out[row] = {
                    name: _display(array[local].as_py()) for name, array in arrays.items()
                }
        return out

    def categorical(
        self, columns: Sequence[str], limit: int = 12, sample_rows: int = 20000
    ) -> dict[str, list[str]]:
        """The columns holding few enough values to offer as a dropdown.

        One read of the file's first row group, sliced, rather than a pass per
        column: at the size this page is for, a row group is large enough that
        reading one per string column would be the slowest thing the page does.
        A column whose first rows already hold more than `limit` values is not
        a dropdown whatever the rest of the file holds, so the slice decides it.
        """
        wanted = [name for name in columns if name in self._file.schema_arrow.names]
        if not wanted or self._file.metadata.num_row_groups == 0:
            return {}
        table = self._file.read_row_group(0, columns=wanted).slice(0, sample_rows)
        found: dict[str, list[str]] = {}
        for name in wanted:
            seen: list[str] = []
            for value in table.column(name).to_pylist():
                text = _display(value)
                if text and text not in seen:
                    seen.append(text)
                    if len(seen) > limit:
                        break
            if seen and len(seen) <= limit:
                found[name] = sorted(seen)
        return found
