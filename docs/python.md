# From Python

The `cpplink` Python package is the same binary, loaded into the interpreter.
It does two things.
`cpplink.run(args)` forwards an argument list to the command line dispatcher, so every example in these pages runs unchanged as `python -m cpplink ...`.
And `cpplink.Linker` loads the parquet input once and runs each stage against the same store, returning the reports as objects rather than text.

One rule scopes the package: no number is computed in Python.
The binding calls the functions the command line calls, through the same pipeline glue, so a model estimated from Python with a seed is byte for byte the model `cpplink estimate --seed` writes.
That is a test, `python/tests/test_parity.py`.

A worked example over a real dataset, with plots and tables at every stage, is the notebook `python/examples/historical_50k.ipynb`.

## Install

The extension links the Arrow C++ the environment holds, which is also the Arrow that `pyarrow` loads in the same process, so the two must be the same build.
Inside the project's conda environment they are, and that is where the package is meant to be built and used:

```sh
conda env update -f environment.yml
conda activate cpplink
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
python -c "import cpplink; print(cpplink.__version__)"
```

The build goes through [scikit-build-core](https://scikit-build-core.readthedocs.io/) and lands in `build/python/`, beside the command line's `build/`.
The editable install redirects the Python sources to `python/cpplink/`, so a change there needs no rebuild; a change to the bindings or the core does, with the same `pip install` line.

!!! note "No wheels"
    A distributable wheel would have to bundle an Arrow matching the ABI of whatever `pyarrow` the user has, which is not a problem this package solves.
    It builds for the environment it is built in.

## The command line, in process

```python
import cpplink

result = cpplink.run(["estimate", "--schema", "schema.json", "--out", "model.json", "sample.parquet"])
result.code, result.stdout, result.stderr
```

`run` takes the arguments the binary would, without the program name, and returns the exit code and both output streams.
`python -m cpplink <command> [options]` forwards `sys.argv` to it and exits with the code.

## The pipeline as a `Linker`

The seven commands of [Getting started](getting-started.md), from Python.
Every method mirrors one command, with the command's flags as keyword arguments, and every report prints the table the command prints.

### 1. Make some data

```python
cpplink.gen_sample("sample.parquet", rows=200000, truth="sample.truth.csv")
```

### 2. Draft a schema and load

```python
schema, draft = cpplink.init("sample.parquet", out="schema.json")
print(draft)                         # what was guessed, column by column

linker = cpplink.Linker(schema, "sample.parquet")
print(linker.inspect())              # cardinality and memory per column
```

`Linker` takes a `Schema` or the path of one, and one parquet file or several.
Several files are several datasets and default to linking; `mode="dedup"` or `"link-and-dedup"` scores the within-file pairs too, exactly as `--mode` does.
`all_pairs=True` is `--all-pairs`.

A `Schema` is edited by dict round trip, which is what `init` itself does:

```python
d = schema.to_dict()
d["blocking"].append({"type": "exact_value", "column": "postcode"})
schema = cpplink.Schema.from_dict(d)
```

### 3. Price the blocking

```python
report = linker.explain_blocking(count=True)
for source in report.sources:
    print(source.name, source.pairs, source.largest_group)
```

### 4. Measure blocking recall

```python
recall = linker.recall("sample.truth.csv", why=True)
recall.metrics.pair_completeness      # the fraction of known pairs the union reaches
recall.misses                         # the MissReport, or None without why=True
print(recall)
```

### 5. Learn the model

```python
model, report = linker.estimate(out="model.json", seed=7)
print(model)                          # m, u and the weight in bits per level
model.lambda_, model.prior_weight()
[(c.name, [(l.label, l.m, l.u) for l in c.levels]) for c in model.comparisons]
```

`estimate` takes every flag of the command by name: `u_sample`, `session_pairs`, `threads`, `iterations`, `lambda_`, `seed`, `fuzzy_u`, `ball_budget`, `tie_holdout`, `tied_bits`, `interactions`, `max_interactions`, `interaction_bits`, and so on.
An `EstimateOptions` can be built and passed as `options=` instead.

### 6. Score the candidates

```python
predict = linker.predict(model, "predictions.parquet", threshold=20)
predict.predictions, predict.enumerated, predict.skipped
```

`out` is a shard directory, or a single `.csv` or `.parquet` file the shards are merged into, and the extension decides.
Give `threshold` in bits or `probability` as a posterior.
The `--no-*` switches are the positive keywords `bounds`, `ceiling`, `signatures`, `ladders` and `interactions`; `fuzzy_tf`, `spill`, `spill_sample`, `threads`, `limit` and `tf_damping` are what they are on the command line.
`verbose=True` prints the plan and writes the progress line to the process's standard error, which a terminal shows and a notebook does not capture.

The predictions are read back through pyarrow, since that file is the pipeline's door out:

```python
import pyarrow.parquet as pq
table = pq.read_table("predictions.parquet")
```

### 7. Join the predictions into clusters

```python
result = linker.cluster("predictions.parquet", truth="sample.truth.csv", out="clusters.csv")
result.quality.f1                     # None without a truth file
result.assignment.root                # numpy uint32, the representative row per row
result.assignment.size                # by representative row; 0 for a non-root
```

The numpy arrays view the C++ vectors and keep the assignment alive for as long as they do.
A `Linker` holds every column, so its `cluster` costs no second load but more memory than the command, which loads the id column alone; `cpplink.cluster_file(schema_path, files, predictions)` is that lighter path.

### And then

```python
linker.explain(id_a, id_b, model=model)      # the levels a pair lands on, and its waterfall
linker.profile()                             # what the columns are worth, before any model
linker.levels()                              # (LevelsReport, Schema with the proposal)
linker.simplify(model, min_gap=1.0)          # (SimplifyReport, Schema or None)
linker.completeness(model)                   # blocking recall with no truth file
linker.rescore(model, "spill/", "again.parquet", threshold=25)
cpplink.merge_predictions("shards/", "merged.csv", schema="schema.json", files=["sample.parquet"])
```

`explain` resolves ids the way the command does, including `dataset:id` where the inputs share ids; `by_row=True` takes row indices.
Its `weight` is the scorer's own, and the parity test asserts it equals the weight in the predictions file.

## Reports

Every report is a bound C++ struct with a read-only attribute per field.
`repr(report)` and `report.text` are the table the command prints, produced by the same printer, so the two cannot disagree.
Where a command has `--json`, the report has `json()`.
Where `estimate` has `--report <file>`, `EstimateReport` has `full_text`: `text` names each session's worst residual pair, `full_text` lists every pair of every session and is what to write to a file.
`cpplink.to_dict(report)` walks the attributes and returns plain Python, recursing into lists and nested reports.

Reports whose printer needs more than the report itself, such as `predict`'s, which prices its zones against the scorer, carry their text from the moment they were made.
That text is the run's, not a recomputation.

## What it costs, and what it does not

The `Linker` releases the GIL for every stage that does work: the load, `estimate`, `predict`, `cluster`, `rescore`, `profile`, `levels`, `simplify`, `recall` and `completeness`.
None of them calls back into Python, so another thread can run while they do.

Errors the core reports as `false` and a message become `cpplink.Error`, a `RuntimeError` carrying the core's own text.

## The cluster viewer

`cpplink-viewer`, the `cpplink_viewer` package in the same wheel, serves the clusters a run produced as a page; `pip install "cpplink[viewer]"` adds DuckDB, its one dependency beyond the binding's.
It reads the files the pipeline wrote and never the compiled module, so it also runs from a checkout with nothing built.
See [The cluster viewer](viewer.md).

## Not in this version

- **In-memory input.** The `Linker` reads parquet files, as the command does. A pyarrow, polars or pandas table through `__arrow_c_stream__` is the next step, and needs the loader split so a table can be appended without a file.
- **In-memory predictions.** `predict` writes a file and the caller reads it back; a per-thread sink handing a `pyarrow.Table` straight out is the same step.
- **A progress callback.** `verbose=True` writes lines to the C-level standard error.
