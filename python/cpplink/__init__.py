# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""cpplink from Python: the command line in process, and a ``Linker`` over it.

Two ways in. ``run(args)`` forwards an argument list to the command line
dispatcher exactly as the binary would receive it. ``Linker`` loads the input
once, a data frame or a parquet file, and runs every stage against the same
store, returning predictions and clusters as frames and the reports as objects
rather than text. Both call the same core; no number is computed in Python.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from . import _cpplink
from ._cpplink import (  # noqa: F401  (re-exported)
    ArrowTable,
    BallOptions,
    BlockingReport,
    CliResult,
    ClusterAssignment,
    ClusterQuality,
    ClusterReport,
    ClusterResult,
    ClusterTable,
    CompletenessOptions,
    CompletenessReport,
    DraftReport,
    Error,
    EstimateOptions,
    EstimateReport,
    Explanation,
    Inspection,
    InteractionOptions,
    LevelsOptions,
    LevelsReport,
    LoadStats,
    MemoryReport,
    MergeReport,
    MissReport,
    Model,
    PairWaterfall,
    PredictionTable,
    PredictReport,
    ProfileOptions,
    ProfileReport,
    RecallMetrics,
    RecallResult,
    RescoreReport,
    Schema,
    SimplifyOptions,
    SimplifyReport,
    __version__,
    known_roles,
    parquet_supported,
    probability_for_weight,
    run,
    weight_for_probability,
)
from ._dict import to_dict

__all__ = [
    "ArrowTable",
    "BallOptions",
    "BlockingReport",
    "ClusterAssignment",
    "ClusterQuality",
    "ClusterReport",
    "ClusterResult",
    "ClusterTable",
    "CliResult",
    "CompletenessOptions",
    "CompletenessReport",
    "DraftReport",
    "Error",
    "EstimateOptions",
    "EstimateReport",
    "Explanation",
    "Inspection",
    "InteractionOptions",
    "LevelsOptions",
    "LevelsReport",
    "Linker",
    "LoadStats",
    "MemoryReport",
    "MergeReport",
    "MissReport",
    "Model",
    "PairWaterfall",
    "PredictReport",
    "PredictionTable",
    "ProfileOptions",
    "ProfileReport",
    "RecallMetrics",
    "RecallResult",
    "RescoreReport",
    "Schema",
    "SimplifyOptions",
    "SimplifyReport",
    "__version__",
    "cluster_file",
    "gen_sample",
    "init",
    "known_roles",
    "merge_predictions",
    "parquet_supported",
    "probability_for_weight",
    "run",
    "to_dict",
    "to_pandas",
    "weight_for_probability",
]

PathLike = str | os.PathLike[str]


def _paths(files: PathLike | Iterable[PathLike]) -> list[str]:
    if isinstance(files, (str, os.PathLike)):
        return [os.fspath(files)]
    return [os.fspath(path) for path in files]


def _path(path: PathLike | None) -> str:
    return "" if path is None else os.fspath(path)


# ---- inputs -----------------------------------------------------------------

# Rows a pandas frame with object columns is converted at a time. Every other
# input reaches the core as the buffers it already holds; an object column is a
# Python object per cell and has to be converted, and slicing bounds that copy
# to one slice rather than the frame.
BATCH_ROWS = 1 << 20

Input = Any  # a frame, a pyarrow table or reader, or anything with __arrow_c_stream__

# What clustering reads of a prediction table; the rest stays where it is.
PREDICTION_COLUMNS = ["dataset_a", "id_a", "dataset_b", "id_b", "match_weight"]


def _is_pandas(obj: Any) -> bool:
    return type(obj).__module__.split(".")[0] == "pandas" and hasattr(obj, "iloc")


def _is_path(obj: Any) -> bool:
    return isinstance(obj, (str, os.PathLike))


def _stream_of(data: Input, columns: list[str], batch_rows: int) -> Any:
    """The object handed to the core: anything with ``__arrow_c_stream__``.

    A pandas frame is exported through pyarrow. Its Arrow-backed and numeric
    columns cross with no copy; an object column (plain Python strings, or a
    list column) is converted one slice of ``batch_rows`` at a time, from a
    generator the core pulls, so the transient is a slice and not a frame. A
    path is opened with pyarrow's parquet reader, which is how a build without
    Arrow reads a file. Everything else is taken as it is.
    """
    if _is_pandas(data):
        import pyarrow as pa

        frame = data
        present = [name for name in columns if name in frame.columns]
        if present:
            frame = frame[present]
        # A numpy object column is a Python object per cell; pandas' own string
        # dtype is not, whatever its `kind` says.
        has_objects = any(
            isinstance(dtype, np.dtype) and dtype.kind == "O" for dtype in frame.dtypes
        )
        if not has_objects or len(frame) <= batch_rows:
            return pa.Table.from_pandas(frame, preserve_index=False)
        # Types are inferred over the whole frame, so a column empty in the first
        # slice is still what the later ones hold.
        schema = pa.Schema.from_pandas(frame, preserve_index=False)

        def slices() -> Any:
            for start in range(0, len(frame), batch_rows):
                # A table, not a batch: an Arrow-backed column may be chunked.
                table = pa.Table.from_pandas(
                    frame.iloc[start : start + batch_rows],
                    schema=schema,
                    preserve_index=False,
                )
                yield from table.to_batches()

        return pa.RecordBatchReader.from_batches(schema, slices())
    if _is_path(data):
        import pyarrow.parquet as pq

        try:
            reader = pq.ParquetFile(os.fspath(data))
        except OSError as failure:
            raise Error(f"cannot open {os.fspath(data)}: {failure}") from failure
        present = [name for name in columns if name in reader.schema_arrow.names]
        batches = reader.iter_batches(columns=present or None)
        schema = reader.schema_arrow
        if present:
            schema = pa_schema_subset(schema, present)
        return _reader_from(schema, batches)
    if hasattr(data, "__arrow_c_stream__"):
        return data
    if hasattr(data, "__arrow_c_array__"):
        import pyarrow as pa

        return pa.RecordBatch.from_arrays(*_arrays_of(data)).to_reader()
    raise TypeError(
        f"{type(data).__name__} is not a pandas frame, a pyarrow table or reader, "
        "a parquet path, or anything else with __arrow_c_stream__"
    )


def pa_schema_subset(schema: Any, names: list[str]) -> Any:
    import pyarrow as pa

    return pa.schema([schema.field(name) for name in names], metadata=schema.metadata)


def _reader_from(schema: Any, batches: Any) -> Any:
    import pyarrow as pa

    return pa.RecordBatchReader.from_batches(schema, batches)


def _arrays_of(data: Any) -> tuple[Any, Any]:
    import pyarrow as pa

    batch = pa.record_batch(data)
    return batch.columns, batch.schema.names


def _parquet_path(path: PathLike | None) -> bool:
    return path is not None and os.fspath(path).lower().endswith((".parquet", ".pq"))


def _core_writes(path: PathLike | None) -> bool:
    """Whether the core writes this output itself. A build without Arrow leaves
    a parquet file to pandas, from the same rows; csv and shards it writes."""
    return not (_parquet_path(path) and not _cpplink.parquet_supported())


def to_pandas(table: Any) -> Any:
    """A :class:`pandas.DataFrame` from a table the core holds.

    The core hands the table over through the Arrow PyCapsule protocol with the
    record ids as a dictionary over the store's own ids, which is what keeps a
    table of fifty million predictions from carrying fifty million strings.
    The frame decodes those to plain string columns, since a pandas
    ``Categorical`` over every record of the store is neither cheap nor what a
    frame of predictions wants; the decode copies only the ids the rows name.
    ``pyarrow.table(table)`` gives the undecoded, zero-copy form.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    arrow = pa.table(table)
    for index, field in enumerate(arrow.schema):
        if pa.types.is_dictionary(field.type):
            arrow = arrow.set_column(
                index, field.name, pc.dictionary_decode(arrow[index])
            )
    return arrow.to_pandas()


def _inputs(
    data: Input | Mapping[str, Input], columns: list[str], batch_rows: int
) -> list[tuple[str, Any]]:
    """``(name, stream)`` per input, in order. A single input has no name."""
    if isinstance(data, Mapping):
        return [
            (str(name), _stream_of(value, columns, batch_rows))
            for name, value in data.items()
        ]
    return [("", _stream_of(data, columns, batch_rows))]


def _apply(options: Any, **values: Any) -> Any:
    """Set the given fields on a bound options object, refusing unknown names."""
    for name, value in values.items():
        if value is None:
            continue
        if not hasattr(options, name):
            raise TypeError(f"unknown option {name!r} for {type(options).__name__}")
        setattr(options, name, value)
    return options


# ---- Schema helpers ---------------------------------------------------------


def _schema_from_dict(cls: type[Schema], data: Mapping[str, Any]) -> Schema:
    return cls.from_json(json.dumps(data, indent=2))


def _schema_to_dict(self: Schema) -> dict[str, Any]:
    return json.loads(self.to_json())


Schema.from_dict = classmethod(_schema_from_dict)  # type: ignore[attr-defined]
Schema.to_dict = _schema_to_dict  # type: ignore[attr-defined]


def _model_to_dict(self: Model) -> dict[str, Any]:
    return json.loads(self.to_json())


Model.to_dict = _model_to_dict  # type: ignore[attr-defined]


def gen_sample(
    out: PathLike | None = None,
    rows: int = 1000000,
    seed: int = 1,
    duplicate_rate: float = 0.08,
    truth: PathLike | None = None,
    out_b: Iterable[PathLike] | int = (),
) -> Any:
    """The sample with planted duplicates, as ``cpplink gen-sample`` makes it.

    With ``out`` the sample is written as parquet and the path returned;
    ``out_b`` then names the later files of a link fixture, one per entry.
    Without ``out`` the rows come back as a :class:`pandas.DataFrame`, or a
    list of frames when ``out_b`` is a count of link files, and no file is
    written but ``truth``, the csv of planted pairs, where it is asked for.
    The same rows either way, from the same generator.
    """
    if out is not None and not isinstance(out_b, int):
        if _cpplink.parquet_supported():
            return _cpplink.gen_sample(
                os.fspath(out), rows, seed, duplicate_rate, _path(truth), _paths(out_b)
            )
        # A build without Arrow makes the tables and lets pandas write them.
        paths = [os.fspath(out), *_paths(out_b)]
        tables = _cpplink.gen_sample_tables(
            rows, seed, duplicate_rate, _path(truth), len(paths) - 1
        )
        for path, table in zip(paths, tables, strict=True):
            to_pandas(table).to_parquet(path, index=False)
        return os.fspath(out)
    link_files = out_b if isinstance(out_b, int) else len(_paths(out_b))
    tables = _cpplink.gen_sample_tables(
        rows, seed, duplicate_rate, _path(truth), link_files
    )
    frames = [to_pandas(table) for table in tables]
    return frames if link_files else frames[0]


def merge_predictions(
    shards: PathLike,
    out: PathLike,
    *,
    format: str | None = None,  # noqa: A002
    source: str = "auto",
    threshold: float | None = None,
    probability: float | None = None,
    batch_rows: int = 65536,
    schema: PathLike | None = None,
    files: Iterable[PathLike] = (),
) -> MergeReport:
    """Fold a shard directory into one csv or parquet file, as ``cpplink
    merge-predictions`` does. ``schema`` and ``files`` together turn the rows
    binary shards name back into ids."""
    return _cpplink.merge_predictions(
        os.fspath(shards),
        os.fspath(out),
        format=format,
        source=source,
        threshold=threshold,
        probability=probability,
        batch_rows=batch_rows,
        schema=_path(schema),
        files=_paths(files),
    )


def cluster_file(
    schema: PathLike,
    files: PathLike | Iterable[PathLike],
    predictions: PathLike,
    *,
    threshold: float | None = None,
    probability: float | None = None,
    out: PathLike | None = None,
    min_size: int = 2,
    truth: PathLike | None = None,
) -> ClusterResult:
    """``cpplink cluster`` as the command runs it: only the id column is loaded.
    A :class:`Linker` already holds every column, so its ``cluster`` costs no
    second load but more memory."""
    return _cpplink.cluster_file(
        os.fspath(schema),
        _paths(files),
        os.fspath(predictions),
        threshold=threshold,
        probability=probability,
        out=_path(out),
        min_size=min_size,
        truth=_path(truth),
    )


def _described(data: Any, name: str) -> tuple[str, Any]:
    """``(name, object with __arrow_c_schema__)`` for one input to draft from."""
    import pyarrow as pa

    if _is_pandas(data):
        return name, pa.Schema.from_pandas(data, preserve_index=False)
    if hasattr(data, "__arrow_c_schema__"):
        return name, data
    if hasattr(data, "__arrow_c_stream__"):
        return name, pa.RecordBatchReader.from_stream(data).schema
    raise TypeError(
        f"{type(data).__name__} is not a pandas frame, a pyarrow table or schema, "
        "or anything else with __arrow_c_schema__"
    )


def init(
    data: PathLike | Iterable[PathLike] | Input | Mapping[str, Input] | None = None,
    id: str | None = None,  # noqa: A002  (the flag's name)
    roles: Mapping[str, str] | None = None,
    out: PathLike | None = None,
    *,
    files: PathLike | Iterable[PathLike] | None = None,
) -> tuple[Schema, DraftReport]:
    """Draft a schema from what the input holds, as ``cpplink init`` does.

    ``data`` is a parquet path or several, whose footers are read; or a
    frame, a pyarrow table or schema, or a ``{name: input}`` dict of them,
    whose Arrow schema says what the footer would have. ``roles`` overrides
    the guessed role of a column, as ``--role COLUMN=ROLE`` does; the accepted
    roles are ``known_roles()``. Returns the schema and the report saying
    what was guessed.
    """
    if files is not None:
        if data is not None:
            raise TypeError("give data or files, not both")
        data = files
    if data is None:
        raise TypeError("init needs data: a path, a frame, a table or a dict of them")
    role_list = [(column, role) for column, role in (roles or {}).items()]
    if _is_path(data) or (
        not isinstance(data, Mapping)
        and isinstance(data, Iterable)
        and not hasattr(data, "__arrow_c_stream__")
        and all(_is_path(item) for item in data)
    ):
        paths = _paths(data)
        if _cpplink.parquet_supported():
            return _cpplink.init(paths, id or "", role_list, _path(out))
        import pyarrow.parquet as pq

        names = _cpplink.dataset_names_for(paths) or [paths[0]]
        inputs = [
            (name, pq.read_schema(path)) for name, path in zip(names, paths, strict=True)
        ]
        return _cpplink.init_from(inputs, id or "", role_list, _path(out))
    if isinstance(data, Mapping):
        inputs = [_described(value, str(name)) for name, value in data.items()]
    else:
        inputs = [_described(data, "the input")]
    return _cpplink.init_from(inputs, id or "", role_list, _path(out))


# ---- Linker -----------------------------------------------------------------


class Linker:
    """One loaded store, and every stage of the pipeline over it.

    ``schema`` is a :class:`Schema` or the path of one. ``data`` is what to
    load: a pandas frame, a pyarrow table or reader, a polars frame or
    anything else with ``__arrow_c_stream__``, or a parquet path; several
    inputs are a ``{name: input}`` dict, read in order into one store with
    each a dataset named by its key, or a list of paths named by their stems.
    ``mode`` is ``"dedup"``, ``"link"`` or ``"link-and-dedup"``, and defaults
    to linking when there is more than one input. ``all_pairs`` replaces the
    schema's blocking with the source that blocks on nothing.

    A frame is read where it sits: its Arrow-backed and numeric columns cross
    into the core with no copy, and an object column is converted
    ``batch_rows`` rows at a time. The store is a re-encoding rather than a
    view, so the frame and the store are both resident while the ``Linker``
    lives; drop the frame if it is not needed again. The frame's index is
    not read: the record id must be a column.

    The estimation and prediction plans are built on first use, from the
    schema's ``"use": "estimate"`` and ``"use": "predict"`` sources
    respectively, which is the same split the command line keeps.
    """

    def __init__(
        self,
        schema: Schema | PathLike,
        data: Input | Mapping[str, Input] | PathLike | Iterable[PathLike] | None = None,
        mode: str | None = None,
        all_pairs: bool = False,
        *,
        files: PathLike | Iterable[PathLike] | None = None,
        batch_rows: int = BATCH_ROWS,
    ) -> None:
        if not isinstance(schema, Schema):
            schema = Schema.from_file(os.fspath(schema))
        self.schema = schema
        self.last_predict: PredictReport | None = None
        self.last_rescore: RescoreReport | None = None
        self.last_cluster: ClusterResult | None = None
        if files is not None:
            if data is not None:
                raise TypeError("give data or files, not both")
            data = files
        if data is None:
            raise TypeError(
                "a Linker needs data: a frame, a table, a path or a dict of them"
            )
        self.files: list[str] = []
        if _is_path(data) or (
            not isinstance(data, Mapping)
            and isinstance(data, Iterable)
            and not hasattr(data, "__arrow_c_stream__")
            and all(_is_path(item) for item in data)
        ):
            self.files = _paths(data)
            if _cpplink.parquet_supported():
                self._session = _cpplink._Session(schema, self.files, mode, all_pairs)
                return
            # A build without Arrow reads the file through pyarrow, under the
            # names the core would have given the files.
            names = _cpplink.dataset_names_for(self.files) or [""]
            if len(self.files) > 1:
                data = dict(zip(names, self.files, strict=True))
            else:
                data = self.files[0]
        inputs = _inputs(data, schema.input_columns, batch_rows)
        self._session = _cpplink._Session.from_streams(schema, inputs, mode, all_pairs)

    # -- what was loaded ------------------------------------------------------

    @property
    def records(self) -> int:
        return self._session.records

    @property
    def datasets(self) -> int:
        return self._session.datasets

    @property
    def mode(self) -> str:
        return self._session.mode

    @property
    def all_pairs(self) -> bool:
        return self._session.all_pairs

    @property
    def stats(self) -> LoadStats:
        return self._session.stats

    def inspect(self) -> Inspection:
        """Cardinality and memory per column, as ``cpplink inspect`` prints."""
        return self._session.inspect()

    def id_of(self, row: int) -> str:
        return self._session.id_of(row)

    def row_of(self, id: str) -> int:  # noqa: A002
        return self._session.row_of(id)

    def __repr__(self) -> str:
        return (
            f"Linker({self.records:,} records, {self.datasets} "
            f"{'dataset' if self.datasets == 1 else 'datasets'}, mode={self.mode})"
        )

    # -- the stages -----------------------------------------------------------

    def estimate(
        self,
        out: PathLike | None = None,
        *,
        u_sample: int | None = None,
        session_pairs: int | None = None,
        threads: int | None = None,
        iterations: int | None = None,
        lambda_: float | None = None,
        seed: int | None = None,
        fuzzy_u: bool = False,
        ball_budget: int | None = None,
        tie_holdout: bool = True,
        tied_bits: float | None = None,
        tie_sample_rows: int | None = None,
        interactions: bool = False,
        max_interactions: int | None = None,
        interaction_bits: float | None = None,
        interaction_clamp: float | None = None,
        min_interaction_sessions: int | None = None,
        min_pairs_per_parameter: float | None = None,
        options: EstimateOptions | None = None,
    ) -> tuple[Model, EstimateReport]:
        """Learn m, u and lambda, as ``cpplink estimate`` does.

        The keywords are the command's flags; ``options`` is an
        :class:`EstimateOptions` to start from instead of the defaults. With
        ``out`` the model is also written there.
        """
        opts = options or EstimateOptions()
        _apply(
            opts,
            u_sample=u_sample,
            session_pairs=session_pairs,
            threads=threads,
            max_iterations=iterations,
            lambda_=lambda_,
            seed=seed,
            tied_bits=tied_bits,
            tie_sample_rows=tie_sample_rows,
        )
        if fuzzy_u:
            opts.fuzzy_u = True
        if ball_budget is not None:
            opts.ball.budget = ball_budget
        if not tie_holdout:
            opts.exclude_tied = False
        if interactions or max_interactions is not None or interaction_bits is not None:
            opts.interactions.enabled = True
        _apply(
            opts.interactions,
            max_terms=max_interactions,
            min_bits=interaction_bits,
            clamp_bits=interaction_clamp,
            min_sessions=min_interaction_sessions,
            min_pairs_per_parameter=min_pairs_per_parameter,
        )
        outcome = self._session.estimate(opts, _path(out))
        return outcome.model, outcome.report

    def predict(
        self,
        model: Model | PathLike,
        out: PathLike | None = None,
        *,
        threshold: float | None = None,
        probability: float | None = None,
        format: str | None = None,  # noqa: A002
        threads: int = 0,
        limit: int = 0,
        tf_damping: float = 1.0,
        bounds: bool = True,
        ceiling: bool = True,
        fuzzy_tf: bool = False,
        ball_budget: int | None = None,
        spill: PathLike | None = None,
        spill_sample: float = 0.0,
        signatures: bool = True,
        ladders: bool = True,
        interactions: bool = True,
        verbose: bool = False,
    ) -> Any:
        """Score the candidate pairs and return the predictions above a threshold.

        Returns a :class:`pandas.DataFrame` with the merged file's columns:
        ``id_a, id_b, gamma, match_weight, match_probability``, with
        ``dataset_a`` and ``dataset_b`` beside the ids over several inputs.
        The run's report is :attr:`last_predict`, and ``last_predict.table``
        is the same rows as an Arrow table, zero-copy. ``out`` also writes
        them as a shard directory or a single ``.csv`` or ``.parquet`` file,
        which a ``rescore`` or the command line needs. Give ``threshold`` in
        bits or a ``probability``. The other keywords are ``cpplink predict``'s
        flags, with ``bounds``, ``ceiling``, ``signatures``, ``ladders`` and
        ``interactions`` the positive forms of its ``--no-*`` switches.
        """
        ball = BallOptions()
        if ball_budget is not None:
            ball.budget = ball_budget
        self.last_predict = self._session.predict(
            _model(model),
            _path(out) if _core_writes(out) else "",
            threshold=threshold,
            probability=probability,
            format=format,
            threads=threads,
            limit=limit,
            tf_damping=tf_damping,
            bounds=bounds,
            ceiling=ceiling,
            fuzzy_tf=fuzzy_tf,
            ball=ball,
            spill=_path(spill),
            spill_sample=spill_sample,
            signatures=signatures,
            ladders=ladders,
            interactions=interactions,
            verbose=verbose,
        )
        frame = to_pandas(self.last_predict.table)
        if not _core_writes(out):
            frame.to_parquet(os.fspath(out), index=False)
        return frame

    def rescore(
        self,
        model: Model | PathLike,
        spill: PathLike,
        out: PathLike | None = None,
        *,
        threshold: float | None = None,
        probability: float | None = None,
        format: str | None = None,  # noqa: A002
        threads: int = 0,
        limit: int = 0,
        tf_damping: float = 1.0,
        bounds: bool = True,
    ) -> Any:
        """Replay a spilled run under a new model, without comparing again.

        Returns the predictions as :meth:`predict` does, a
        :class:`pandas.DataFrame`, with the report on :attr:`last_rescore`;
        ``out`` writes them as well.
        """
        self.last_rescore = self._session.rescore(
            _model(model),
            os.fspath(spill),
            _path(out) if _core_writes(out) else "",
            threshold=threshold,
            probability=probability,
            format=format,
            threads=threads,
            limit=limit,
            tf_damping=tf_damping,
            bounds=bounds,
        )
        frame = to_pandas(self.last_rescore.table)
        if not _core_writes(out):
            frame.to_parquet(os.fspath(out), index=False)
        return frame

    def cluster(
        self,
        predictions: Any = None,
        *,
        threshold: float | None = None,
        probability: float | None = None,
        out: PathLike | None = None,
        min_size: int = 2,
        truth: PathLike | None = None,
    ) -> Any:
        """Join the predictions into clusters with a union-find.

        Returns a :class:`pandas.DataFrame` with the cluster file's columns:
        ``unique_id, cluster_id, cluster_size`` for every record in a cluster
        of ``min_size`` or more, with ``dataset`` first over several inputs.
        ``predictions`` is ``None`` for the rows the last :meth:`predict`
        left in memory, which need no id lookup; a frame ``predict`` returned,
        or any table with ``id_a``, ``id_b`` and ``match_weight`` columns; or
        the shard directory or merged file a run wrote. The partition, its
        report and, with ``truth``, its quality are :attr:`last_cluster`. A
        ``Linker`` holds every column; to cluster with only the id column
        loaded, as the command does, use :func:`cluster_file`.
        """
        if predictions is None:
            source = None
        elif _is_path(predictions):
            source = os.fspath(predictions)
            if _parquet_path(source) and not _cpplink.parquet_supported():
                # A build without Arrow reads the merged file through pyarrow.
                source = _stream_of(source, PREDICTION_COLUMNS, BATCH_ROWS)
        else:
            source = _stream_of(predictions, PREDICTION_COLUMNS, BATCH_ROWS)
        self.last_cluster = self._session.cluster(
            source,
            threshold=threshold,
            probability=probability,
            out=_path(out) if _core_writes(out) else "",
            min_size=min_size,
            truth=_path(truth),
        )
        frame = to_pandas(self.last_cluster.table)
        if not _core_writes(out):
            frame.to_parquet(os.fspath(out), index=False)
        return frame

    def explain(
        self,
        a: str | int,
        b: str | int,
        *,
        model: Model | PathLike | None = None,
        by_row: bool = False,
        threshold: float = 0.0,
        tf_damping: float = 1.0,
        fuzzy_tf: bool = False,
        ball_budget: int | None = None,
        interactions: bool = True,
    ) -> Explanation:
        """The level each comparison lands on for one pair, and with a model
        the waterfall of bits behind its score.

        ``a`` and ``b`` are record ids (``dataset:id`` where the inputs share
        ids), or row indices with ``by_row=True``.
        """
        if by_row:
            row_a, row_b = int(a), int(b)
        else:
            row_a, row_b = self._session.row_of(str(a)), self._session.row_of(str(b))
        ball = BallOptions()
        if ball_budget is not None:
            ball.budget = ball_budget
        return self._session.explain_rows(
            row_a,
            row_b,
            model=None if model is None else _model(model),
            threshold=threshold,
            tf_damping=tf_damping,
            fuzzy_tf=fuzzy_tf,
            ball=ball,
            interactions=interactions,
        )

    def explain_blocking(self, count: bool = False) -> BlockingReport:
        """Price every blocking source without enumerating a pair; ``count``
        also enumerates the deduplicated union."""
        return self._session.explain_blocking(count)

    def recall(
        self,
        truth: PathLike,
        *,
        why: bool = False,
        count: bool = False,
        show_misses: int = 0,
    ) -> RecallResult:
        """What fraction of the known pairs blocking reaches, and with ``why``
        a diagnosis of the ones it does not."""
        return self._session.recall(
            os.fspath(truth), why=why, count=count, show_misses=show_misses
        )

    def profile(
        self,
        *,
        truth: PathLike | None = None,
        sample_rows: int | None = None,
        pairs: bool = True,
        expected_matches: int | None = None,
        threads: int | None = None,
        seed: int | None = None,
        anchors: bool = True,
        anchor_rows: int | None = None,
        anchor_margin: float | None = None,
        anchor_pairs: int | None = None,
        options: ProfileOptions | None = None,
    ) -> ProfileReport:
        """What the columns can be worth, what a match will score, and which
        pairs of them are the same evidence twice."""
        opts = options or ProfileOptions()
        _apply(
            opts,
            sample_rows=sample_rows,
            expected_matches=expected_matches,
            threads=threads,
            seed=seed,
            anchor_rows=anchor_rows,
            anchor_margin=anchor_margin,
            anchor_pairs=anchor_pairs,
        )
        if not pairs:
            opts.pairs = False
        if not anchors:
            opts.anchors = False
        return self._session.profile(opts, _path(truth))

    def levels(
        self,
        *,
        truth: PathLike | None = None,
        out: PathLike | None = None,
        levels: int | None = None,
        max_levels: int | None = None,
        jaro_floor: float | None = None,
        jaro_step: float | None = None,
        edit_max: int | None = None,
        min_match_pairs: int | None = None,
        anchor_margin: float | None = None,
        anchor_rows: int | None = None,
        anchor_pairs: int | None = None,
        expected_matches: int | None = None,
        ball_budget: int | None = None,
        threads: int | None = None,
        options: LevelsOptions | None = None,
    ) -> tuple[LevelsReport, Schema]:
        """Check the fuzzy thresholds against the columns they run on and propose
        better ones. Returns the report and the schema with the proposal written
        in; with ``out`` the schema is also written there."""
        opts = options or LevelsOptions()
        _apply(
            opts,
            levels=levels,
            max_levels=max_levels,
            jaro_floor=jaro_floor,
            jaro_step=jaro_step,
            edit_max=edit_max,
            min_match_pairs=min_match_pairs,
        )
        _apply(
            opts.profile,
            anchor_margin=anchor_margin,
            anchor_rows=anchor_rows,
            anchor_pairs=anchor_pairs,
            expected_matches=expected_matches,
        )
        if ball_budget is not None:
            opts.ball.budget = ball_budget
        if threads is not None:
            opts.threads = threads
            opts.profile.threads = threads
        report = self._session.levels(opts, _path(truth))
        proposal = Schema.from_json(_cpplink._rewrite_schema(self.schema.text, report))
        if out is not None:
            proposal.save(os.fspath(out))
        return report, proposal

    def simplify(
        self,
        model: Model | PathLike,
        *,
        out: PathLike | None = None,
        alpha: float | None = None,
        min_gap: float | None = None,
        pair_cap: int | None = None,
        threads: int | None = None,
        seed: int | None = None,
        options: SimplifyOptions | None = None,
    ) -> tuple[SimplifyReport, Schema | None]:
        """Merge the adjacent levels a scored stream cannot tell apart. Returns
        the report and the simplified schema, or None where nothing merged."""
        opts = options or SimplifyOptions()
        _apply(
            opts,
            alpha=alpha,
            min_gap=min_gap,
            pair_cap=pair_cap,
            threads=threads,
            seed=seed,
        )
        report = self._session.simplify(_model(model), opts)
        if report.merged_levels == 0:
            return report, None
        proposal = Schema.from_json(_cpplink._rewrite_schema(self.schema.text, report))
        if out is not None:
            proposal.save(os.fspath(out))
        return report, proposal

    def completeness(
        self,
        model: Model | PathLike,
        *,
        truth: PathLike | None = None,
        threads: int | None = None,
        sample: float | None = None,
        seed: int | None = None,
        value_weighting: str | None = None,
        min_observed: int | None = None,
        bound_only: bool = False,
        options: CompletenessOptions | None = None,
    ) -> CompletenessReport:
        """Estimate blocking recall with no known pairs at all."""
        opts = options or CompletenessOptions()
        _apply(opts, threads=threads, sample=sample, seed=seed, min_observed=min_observed)
        if value_weighting is not None:
            if value_weighting not in ("records", "pairs"):
                raise ValueError('value_weighting wants "records" or "pairs"')
            opts.pair_weighting = value_weighting == "pairs"
        if bound_only:
            opts.skip_observed = True
        return self._session.completeness(_model(model), opts, _path(truth))


def _model(model: Model | PathLike) -> Model:
    if isinstance(model, Model):
        return model
    return Model.from_file(os.fspath(model))
