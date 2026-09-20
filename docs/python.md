# From Python

The `cpplink` Python package is the same binary, loaded into the interpreter.
It does two things.
`cpplink.run(args)` forwards an argument list to the command line dispatcher, so every example in these pages runs unchanged as `python -m cpplink ...`.
And `cpplink.Linker` loads the input once, a pandas frame or a parquet file, and runs each stage against the same store, returning predictions and clusters as frames and the reports as objects rather than text.

One rule scopes the package: no number is computed in Python.
The binding calls the functions the command line calls, through the same pipeline glue, so a model estimated from Python with a seed is byte for byte the model `cpplink estimate --seed` writes.
That is a test, `python/tests/test_parity.py`.

A worked example over a real dataset, with plots and tables at every stage, is the notebook `python/examples/historical_50k.ipynb`.

## Install

Inside the project's conda environment:

```sh
conda env update -f environment.yml
conda activate cpplink
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
python -c "import cpplink; print(cpplink.__version__)"
```

The build goes through [scikit-build-core](https://scikit-build-core.readthedocs.io/) and lands in `build/python/`, beside the command line's `build/`.
The editable install redirects the Python sources to `python/cpplink/`, so a change there needs no rebuild; a change to the bindings or the core does, with the same `pip install` line.

!!! note "What the module links"
    Nothing but the C++ standard library.
    Every frame, table and file crosses between Python and the core through the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), three C structs with a frozen ABI that the core vendors, so no Arrow C++ object and no `pyarrow` version is part of the contract.
    `pyarrow` is what pandas uses to export a frame and what reads a parquet file into the module, and any version of it will do.
    The conda build above links Arrow C++ as well, so that `predict(out="x.parquet")` and the command line write parquet themselves; a wheel built with `-DCPPLINK_WITH_ARROW=OFF` leaves that to pandas.

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
frame = cpplink.gen_sample(rows=200000, truth="sample.truth.csv")   # a pandas frame
cpplink.gen_sample("sample.parquet", rows=200000)                   # or the file
```

### 2. Draft a schema and load

```python
schema, draft = cpplink.init(frame, out="schema.json")
print(draft)                         # what was guessed, column by column

linker = cpplink.Linker(schema, frame)
print(linker.inspect())              # cardinality and memory per column
```

`Linker` takes a `Schema` or the path of one, and the data: a pandas frame, a pyarrow table or reader, a polars frame or anything else with `__arrow_c_stream__`, or a parquet path.
Several inputs are a `{name: input}` dict, each becoming a dataset named by its key, or a list of paths named by their stems; two inputs and no `mode` means linking them.

```python
linker = cpplink.Linker(schema, {"left": customers, "right": orders})
```

A frame is read where it sits.
Its Arrow-backed and numeric columns cross into the core with no copy; a column of Python objects (plain strings, or a list column) is converted `batch_rows` rows at a time, so the transient is one slice and not a second frame.
A `category` column is the store's own shape and costs one intern per distinct value.
The store is a re-encoding, not a view: the frame and the store are both resident while the `Linker` lives, so drop the frame when it is not needed again.
The frame's index is not read; the record id must be a column, and an integer id is read as its digits.
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
predictions = linker.predict(model, threshold=20)       # a pandas frame
report = linker.last_predict
report.predictions, report.enumerated, report.skipped
```

The frame has the merged file's columns, `id_a, id_b, gamma, match_weight, match_probability`, with `dataset_a` and `dataset_b` beside the ids over several inputs.
`out=` also writes the run as a shard directory or a single `.csv` or `.parquet` file, which `rescore` and the command line read; the extension decides.
Give `threshold` in bits or `probability` as a posterior.
The `--no-*` switches are the positive keywords `bounds`, `ceiling`, `signatures`, `ladders` and `interactions`; `fuzzy_tf`, `spill`, `spill_sample`, `threads`, `limit` and `tf_damping` are what they are on the command line.
`verbose=True` prints the plan and writes the progress line to the process's standard error, which a terminal shows and a notebook does not capture.

`report.table` is the same rows as the core holds them, readable by anything that speaks the Arrow PyCapsule protocol and copied by none of it:

```python
import pyarrow as pa
table = pa.table(report.table)     # zero copy; the ids are a dictionary over the store's ids
```

The frame decodes that dictionary into plain string columns, since a pandas `Categorical` over every record of the store is neither cheap nor what a frame of predictions wants; the decode copies only the ids the rows name.

### 7. Join the predictions into clusters

```python
clusters = linker.cluster(truth="sample.truth.csv")    # a pandas frame
result = linker.last_cluster
result.quality.f1                     # None without a truth file
result.assignment.root                # numpy uint32, the representative row per row
result.assignment.size                # by representative row; 0 for a non-root
```

The frame has the cluster file's columns, `unique_id, cluster_id, cluster_size`, one row per record in a cluster of `min_size` or more, with `dataset` first over several inputs.
With nothing given, `cluster` reads the rows the last `predict` left in memory, which name rows and need no id lookup.
It also takes the frame `predict` returned, or any table with `id_a`, `id_b` and `match_weight` columns, or the shard directory or merged file a run wrote; those name records by id and cost the index the command pays.
`out=` writes the frame as `.csv` or `.parquet` too.
Which member names a cluster depends on the order the edges arrived, so two runs agree on the partition and not always on the `cluster_id`.

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

## Not in this version

- **A progress callback.** `verbose=True` writes lines to the C-level standard error.
- **Streaming output.** `predict` holds every prediction in memory, twenty bytes each, before the frame is made; a run whose predictions do not fit writes shards with `out=` instead.
