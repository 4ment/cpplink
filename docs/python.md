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

Inside the project's conda environment, against the Arrow C++ it holds:

```sh
conda env update -f environment.yml
conda activate cpplink
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
python -c "import cpplink; print(cpplink.__version__)"
```

The build goes through [scikit-build-core](https://scikit-build-core.readthedocs.io/) and lands in `build/python/`, beside the command line's `build/`.
The editable install redirects the Python sources to `python/cpplink/`, so a change there needs no rebuild; a change to the bindings or the core does, with the same `pip install` line.

Outside conda, in any environment with a C++17 compiler, the package builds with no Arrow at all:

```sh
pip install . -Ccmake.define.CPPLINK_WITH_ARROW=OFF
```

That is the build the wheels workflow makes, and it links nothing but the C++ standard library: `pandas`, `numpy` and `pyarrow` are its only dependencies, at whatever versions the environment holds.
`cpplink.parquet_supported()` says which build is loaded.

!!! note "What the module links"
    Every frame, table and file crosses between Python and the core through the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), three C structs with a frozen ABI that the core vendors, so no Arrow C++ object and no `pyarrow` version is part of the contract.
    `pyarrow` is what pandas uses to export a frame and what reads a parquet file into the module, and any version of it will do.
    The conda build links Arrow C++ as well, so that `predict(out="x.parquet")` and the command line write parquet themselves.

What a build without Arrow does differently, all of it at the edges:

- A parquet path given to `Linker`, `init`, `cluster` or `gen_sample` is read or written through `pyarrow` and `pandas`, from the same rows the conda build reads and writes itself, so the model, the predictions and the clusters are the same.
- `cpplink.run` and `python -m cpplink` still dispatch every command, but the ones that open a parquet file fail with the reason, since that is the core reading the file.
- `cluster_file`, which loads the id column alone, and `merge_predictions` over binary shards, which needs the input to turn rows back into ids, need the core to read parquet and are refused the same way.

The test suite marks those paths `needs_core_parquet` and skips them there; everything else runs on both builds.

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

`seed` and `duplicate_rate` are the command's flags.
`out_b` makes the link fixture, originals in the first file and every planted duplicate in a later one: a list of paths with `out`, or a count of later files without it, which then returns a list of frames.

```python
cpplink.gen_sample("a.parquet", rows=20000, truth="link.truth.csv", out_b=["b.parquet"])
a, b = cpplink.gen_sample(rows=20000, out_b=1)
```

### 2. Draft a schema and load

```python
schema, draft = cpplink.init(frame, out="schema.json")
print(draft)                         # what was guessed, column by column

linker = cpplink.Linker(schema, frame)
print(linker.inspect())              # cardinality and memory per column
```

`init` reads no rows: a parquet path gives up its footer, and a frame, a pyarrow table or schema, or a `{name: input}` dict of them gives up its Arrow schema, which says what the footer would have.
`roles={"birth_place": "city"}` overrides a guessed role, as `--role COLUMN=ROLE` does, and `id=` names the record id; the roles `init` accepts are `cpplink.known_roles()`.
The report has a `DraftColumn` per column, with its `type`, its `role` and whether the role was `role_given` or guessed.

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

What was loaded is on the `Linker` itself: `records`, `datasets`, `mode` (the pairs the store admits: `all pairs` deduplicating, `cross-dataset pairs` linking), `all_pairs`, `files` and `stats`, the `LoadStats` with the rows and seconds per input.
`id_of(row)` and `row_of(id)` go between a row index and a record id, and `repr(linker)` says what it holds.

```python
>>> linker
Linker(200,000 records, 1 dataset, mode=all pairs)
```

A `Schema` comes from `Schema.from_file`, `Schema.from_json` or `Schema.from_dict`, and is edited by dict round trip, which is what `init` itself does:

```python
d = schema.to_dict()
d["blocking"].append({"type": "exact_value", "column": "postcode"})
schema = cpplink.Schema.from_dict(d)
schema.save("schema.json")
```

`to_json` is the writer `cpplink init` uses, and `save` writes the source text where the schema came from one.
The read-only side describes what was parsed: `unique_id`, `column_names`, `input_columns` (the id and every column that is not derived, which is what a reader is asked for), `comparison_names`, `comparisons` (the description of each level, per comparison), `blocking` and `gamma_width`.

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

`estimate` takes every flag of the command by name: `u_sample`, `session_pairs`, `threads`, `iterations`, `lambda_`, `seed`, `fuzzy_u`, `ball_budget`, `tie_holdout`, `tied_bits`, `tie_sample_rows`, `interactions`, `max_interactions`, `interaction_bits`, `interaction_clamp`, `min_interaction_sessions` and `min_pairs_per_parameter`.
An `EstimateOptions` can be built and passed as `options=` instead, with the keywords applied over it.

A `Model` is the `model.json` file as an object: `Model.from_file` and `Model.from_json` read one, `to_json`, `to_dict` and `save` write it, and `text` or `repr(model)` is the table `estimate` prints.
It carries `lambda_`, `lambda_basis`, `records`, a `ModelComparison` per comparison with a `ModelLevel` per level (`label`, `m`, `u` and `weight()`, the bits `log2(m / u)`), and a `ModelInteraction` per fitted two-way term.
Every method that takes a `model` takes a `Model` or the path of one.

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
`cpplink.to_pandas(table)` is that decode, for any table the core hands out.

`cpplink.weight_for_probability` and `cpplink.probability_for_weight` are the two conversions the reports use, so a threshold can be stated either way and read back:

```python
>>> cpplink.probability_for_weight(20)
0.9999990463265931
>>> cpplink.weight_for_probability(0.99)
6.629356620079609
```

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

### 8. Search for a record

A resident `Linker` is a search service: one store loaded once, a query answered in milliseconds.

```python
hits = linker.search(
    {"last_name": "zolnerowich", "dob": "1979-08-06", "postcode": "4508"},
    model,
    k=10,
    expected_matches=1,
)
for hit in hits:
    print(hit.id, hit.match_weight, hit.match_probability)
hits.report.walk_seconds, hits.report.gather_seconds   # what each phase cost
```

The query maps column names to values, as text, in the form the file holds them; a date is `YYYY-MM-DD` and is refused in any other form.
A list or a tuple is spread over the repeats a list column takes one element per, and `None` is the same as leaving the column out.
A column the query does not name is **missing**, not empty, so its comparison lands on its null level and costs nothing at all.

The answer is exact with respect to the model: the same `k` records, in the same order, as scoring the query against every record.
`SearchResult` iterates its hits, indexes them and carries the printed report as `text`, the numbers as `report` and the command's `--json` as `json()`.

`expected_matches` is the prior stated as the number of records here you expect to be the person asked about, which is the question a search asks; λ is the match rate over the *pair* space, which is the question deduplication asks, and the two differ by orders of magnitude.
Without it the model's own prior applies and a hit scores exactly what `predict` would have given that pair.
The prior is a constant added to every hit, so it moves the posterior and where a threshold sits, and never the order.

```python
hits = linker.search({"last_name": "zolnerowich"}, model, k=3, explain=True)
print(hits.waterfalls[0])        # the same ledger `explain` prints, for that pair
```

`explain=True` builds the ledger behind each hit during the search, which is when it has to happen: the query is a record of the store only while the search runs, so a result that outlived it would be a result nothing could explain.
`clusters=` takes the file `cluster` wrote and labels each hit with the cluster it belongs to, marking the ones that are another record of an entity already listed.

Two limits, both narrow.
`search` reads one input and raises over several, because the query row would have to be a dataset of its own.
And one search runs at a time against one store; `threads` splits the work *within* a query rather than running several.

### And then

```python
linker.explain(id_a, id_b, model=model)      # the levels a pair lands on, and its waterfall
linker.profile()                             # what the columns are worth, before any model
linker.levels()                              # (LevelsReport, Schema with the proposal)
linker.simplify(model, min_gap=1.0)          # (SimplifyReport, Schema or None)
linker.completeness(model)                   # blocking recall with no truth file
linker.rescore(model, "spill/", threshold=25)  # a frame, with the report on last_rescore
cpplink.merge_predictions("shards/", "merged.csv", schema="schema.json", files=["sample.parquet"])
cpplink.cluster_file("schema.json", "sample.parquet", "predictions.parquet", truth="sample.truth.csv")
```

`explain` resolves ids the way the command does, including `dataset:id` where the inputs share ids; `by_row=True` takes row indices.
The `Explanation` has `levels`, comparison name to the label the pair landed on, and with a model a `waterfall`, the `PairWaterfall` ledger of `prior`, one `WaterfallStep` per comparison with its `bits`, `tf` move and `running` total, the `interactions`, and the final `weight` and `probability`; `json()` is the command's `--json`.
Its `weight` is the scorer's own, and the parity test asserts it equals the weight in the predictions file.
`explain` takes `fuzzy_tf` and `interactions=False` for the same reason the command does: scored under different options than the run, it explains a different weight than the file holds.

`rescore` replays a spill `predict` wrote under a new model and returns the predictions as `predict` does, with `out=` writing them the same two ways.
`levels` and `simplify` return the schema with the proposal written in, or `None` from `simplify` where nothing merged, and `out=` writes it.

## Reports

Every report is a bound C++ struct with a read-only attribute per field.
`repr(report)` and `report.text` are the table the command prints, produced by the same printer, so the two cannot disagree.
Where a command has `--json`, the report has `json()`.
Where `estimate` has `--report <file>`, `EstimateReport` has `full_text`: `text` names each session's worst residual pair, `full_text` lists every pair of every session and is what to write to a file.
`cpplink.to_dict(report)` walks the attributes and returns plain Python, recursing into lists and nested reports.

Reports whose printer needs more than the report itself, such as `predict`'s, which prices its zones against the scorer, carry their text from the moment they were made.
That text is the run's, not a recomputation.

## What it costs, and what it does not

The `Linker` releases the GIL for every stage that does work: the load, `estimate`, `predict`, `cluster`, `rescore`, `profile`, `levels`, `simplify`, `recall`, `completeness` and `search`.
None of them calls back into Python, so another thread can run while they do.

Errors the core reports as `false` and a message become `cpplink.Error`, a `RuntimeError` carrying the core's own text.

## The cluster viewer

`cpplink-viewer`, the `cpplink_viewer` package in the same wheel, serves the clusters a run produced as a page; `pip install "cpplink[viewer]"` adds DuckDB, its one dependency beyond the binding's.
It reads the files the pipeline wrote and never the compiled module, so it also runs from a checkout with nothing built.
See [The cluster viewer](viewer.md).

## Not in this version

- **A progress callback.** `verbose=True` writes lines to the C-level standard error.
- **Streaming output.** `predict` holds every prediction in memory, twenty bytes each, before the frame is made; a run whose predictions do not fit writes shards with `out=` instead.
