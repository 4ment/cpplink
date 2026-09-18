# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""cpplink from Python: the command line in process, and a ``Linker`` over it.

Two ways in. ``run(args)`` forwards an argument list to the command line
dispatcher exactly as the binary would receive it. ``Linker`` loads the parquet
input once and runs every stage against the same store, returning the reports
as objects rather than text. Both call the same core; no number is computed in
Python.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Sequence

from . import _cpplink
from ._cpplink import (  # noqa: F401  (re-exported)
    BallOptions,
    BlockingReport,
    ClusterAssignment,
    ClusterQuality,
    ClusterReport,
    ClusterResult,
    CliResult,
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
    probability_for_weight,
    run,
    weight_for_probability,
)
from ._dict import to_dict

__all__ = [
    "BallOptions",
    "BlockingReport",
    "ClusterAssignment",
    "ClusterQuality",
    "ClusterReport",
    "ClusterResult",
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
    "probability_for_weight",
    "run",
    "to_dict",
    "weight_for_probability",
]

PathLike = str | os.PathLike[str]


def _paths(files: PathLike | Iterable[PathLike]) -> list[str]:
    if isinstance(files, (str, os.PathLike)):
        return [os.fspath(files)]
    return [os.fspath(path) for path in files]


def _path(path: PathLike | None) -> str:
    return "" if path is None else os.fspath(path)


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
    out: PathLike,
    rows: int = 1000000,
    seed: int = 1,
    duplicate_rate: float = 0.08,
    truth: PathLike | None = None,
    out_b: Iterable[PathLike] = (),
) -> str:
    """Write a sample parquet file with planted duplicates, as ``cpplink
    gen-sample`` does. ``out_b`` names the later files of a link fixture, one
    file per entry, and ``truth`` the csv of planted pairs."""
    return _cpplink.gen_sample(
        os.fspath(out), rows, seed, duplicate_rate, _path(truth), _paths(out_b)
    )


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


def init(
    files: PathLike | Iterable[PathLike],
    id: str | None = None,  # noqa: A002  (the flag's name)
    roles: Mapping[str, str] | None = None,
    out: PathLike | None = None,
) -> tuple[Schema, DraftReport]:
    """Draft a schema from a parquet footer, as ``cpplink init`` does.

    ``roles`` overrides the guessed role of a column, as ``--role COLUMN=ROLE``
    does; the accepted roles are ``known_roles()``. Returns the schema and the
    report saying what was guessed.
    """
    role_list = [(column, role) for column, role in (roles or {}).items()]
    return _cpplink.init(_paths(files), id or "", role_list, _path(out))


# ---- Linker -----------------------------------------------------------------


class Linker:
    """One loaded store, and every stage of the pipeline over it.

    ``schema`` is a :class:`Schema` or the path of one; ``files`` one parquet
    path or several, read in order into one store with each file a dataset.
    ``mode`` is ``"dedup"``, ``"link"`` or ``"link-and-dedup"``, and defaults to
    linking when there is more than one file. ``all_pairs`` replaces the
    schema's blocking with the source that blocks on nothing.

    The estimation and prediction plans are built on first use, from the
    schema's ``"use": "estimate"`` and ``"use": "predict"`` sources
    respectively, which is the same split the command line keeps.
    """

    def __init__(
        self,
        schema: Schema | PathLike,
        files: PathLike | Iterable[PathLike],
        mode: str | None = None,
        all_pairs: bool = False,
    ) -> None:
        if not isinstance(schema, Schema):
            schema = Schema.from_file(os.fspath(schema))
        self.schema = schema
        self.files = _paths(files)
        self._session = _cpplink._Session(schema, self.files, mode, all_pairs)

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
        out: PathLike,
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
    ) -> PredictReport:
        """Score the candidate pairs and write the predictions above a threshold.

        ``out`` is a shard directory, or a single ``.csv`` or ``.parquet`` file
        the shards are merged into. Give ``threshold`` in bits or a
        ``probability``. The other keywords are ``cpplink predict``'s flags,
        with ``bounds``, ``ceiling``, ``signatures``, ``ladders`` and
        ``interactions`` the positive forms of its ``--no-*`` switches.
        """
        ball = BallOptions()
        if ball_budget is not None:
            ball.budget = ball_budget
        return self._session.predict(
            _model(model),
            os.fspath(out),
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

    def rescore(
        self,
        model: Model | PathLike,
        spill: PathLike,
        out: PathLike,
        *,
        threshold: float | None = None,
        probability: float | None = None,
        format: str | None = None,  # noqa: A002
        threads: int = 0,
        limit: int = 0,
        tf_damping: float = 1.0,
        bounds: bool = True,
    ) -> RescoreReport:
        """Replay a spilled run under a new model, without comparing again."""
        return self._session.rescore(
            _model(model),
            os.fspath(spill),
            os.fspath(out),
            threshold=threshold,
            probability=probability,
            format=format,
            threads=threads,
            limit=limit,
            tf_damping=tf_damping,
            bounds=bounds,
        )

    def cluster(
        self,
        predictions: PathLike,
        *,
        threshold: float | None = None,
        probability: float | None = None,
        out: PathLike | None = None,
        min_size: int = 2,
        truth: PathLike | None = None,
    ) -> ClusterResult:
        """Join the predictions into clusters with a union-find.

        ``predictions`` is the shard directory or the merged file ``predict``
        wrote. With ``truth`` the partition is scored against the known pairs
        and ``result.quality`` is set. A ``Linker`` holds every column; to
        cluster with only the id column loaded, as the command does, use
        :func:`cluster_file`.
        """
        return self._session.cluster(
            os.fspath(predictions),
            threshold=threshold,
            probability=probability,
            out=_path(out),
            min_size=min_size,
            truth=_path(truth),
        )

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
            opts, alpha=alpha, min_gap=min_gap, pair_cap=pair_cap, threads=threads, seed=seed
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
