# Getting started

## Prerequisites

cpplink needs CMake, a C++17 compiler, Arrow C++ (`libarrow` / `libparquet`),
`nlohmann_json`, and GoogleTest for the test suite. All of them except the compiler are
declared in [`environment.yml`](https://github.com/4ment/cpplink/blob/main/environment.yml):

```sh
conda env create -f environment.yml
conda activate cpplink
```

The environment deliberately ships no compiler — builds use the system `clang++`.

## Build

```sh
cmake -S . -B build
cmake --build build
```

With the test suite:

```sh
cmake -S . -B build -DBUILD_TESTING=on -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
ctest --test-dir build
```

`-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` is what makes `find_package(GTest)` succeed; without it
the test configure step fails even inside the activated environment.

The presets in `CMakePresets.json` hold the same configuration, plus two instrumented ones: `asan` builds with AddressSanitizer and UndefinedBehaviorSanitizer, `tsan` with ThreadSanitizer, each into its own `build-<preset>/`.
The threaded stages (the histogram fold, `predict`, `rescore` and the neighbourhood self-join) share the record store `const` across threads with no lock, and the `tsan` preset is what checks that claim rather than asserting it.
Arrow and Parquet are not instrumented, so `tests/sanitizer_suppressions.cpp` compiles a `called_from_lib` suppression for the two libraries into the test binary: their own IO threads would otherwise report a race the sanitizer cannot see the ordering of, while every access cpplink makes stays checked.

```sh
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

`release` and `debug` are the plain builds with tests on, and `ctest --preset <name>` runs the suite with output on failure.
On Windows the `windows` preset builds with Visual Studio 2026 against the conda-forge Arrow, which lives under `%CONDA_PREFIX%\Library`, and the build and test presets of the same name select the `Release` configuration, since the Visual Studio generator holds every configuration in one tree.
The GitHub Actions workflow runs `release`, `asan` and `tsan` on Linux and macOS and `windows` on Windows, then the Python suite and the format and lint checks.

## The whole pipeline in eight commands

Everything below runs against the synthetic sample cpplink can write for itself. The numbers
shown are from a real run on 1.8M rows with eight threads.

### 1. Make some data

```sh
./build/cpplink gen-sample --out examples/sample.parquet --rows 1800000 \
                           --truth examples/sample.truth.csv
```

8% of the rows are corrupted copies of earlier rows, and `--truth` records which. That
sidecar is the ground truth [`recall`](commands/recall.md) and [`cluster`](commands/cluster.md)
score against.

### 2. Look at what loaded

```sh
./build/cpplink inspect --schema examples/sample_schema.json examples/sample.parquet
```

```text
Records      1,800,000
Row groups   9
Load time    2.6 s  (691,485 rows/s)

Column            Type                Distinct      Null   Top val Mean len        Rare pairs
---------------------------------------------------------------------------------------------
first_name        string                23,108     0.00%     2.27%        -            52,587
last_name         string               173,313     0.00%     0.20%        -        16,659,085
dob               date                  20,821     0.00%     0.01%        -        68,281,433
...
Resident total                        332.8 MB   193 bytes per record
```

See [`inspect`](commands/inspect.md).

### 3. Price the blocking before you pay for it

```sh
./build/cpplink explain-blocking --schema examples/sample_schema.json --count \
                                 examples/sample.parquet
```

```text
Pairs unblocked  1,619,999,100,000
...
Sum over sources                                  130,859,994
Union, deduplicated                               116,939,057
Blocking keeps 8.08e-05 of all possible pairs.
```

The counts are exact and come from the term-frequency tables — not one pair is enumerated.
See [`explain-blocking`](commands/explain-blocking.md).

### 4. Measure what blocking misses

```sh
./build/cpplink recall --schema examples/sample_schema.json \
                       --truth examples/sample.truth.csv examples/sample.parquet
```

```text
Union                          142,657  99.25%     130,859,994    0.109%

1071 known pairs are reachable by no source.
```

This is the ceiling on everything downstream: a pair that is never a candidate cannot be
scored. See [`recall`](commands/recall.md).

### 5. Learn the model

```sh
./build/cpplink estimate --schema examples/sample_schema.json \
                         --out model.json examples/sample.parquet
```

Four EM sessions, 242–553 distinct patterns each, 2–10 iterations, about 7 s of work. The
printed table is the whole model: `m`, `u` and the weight in bits for every level. See
[`estimate`](commands/estimate.md) and [Estimation and EM](em.md).

### 6. Score the candidates

```sh
./build/cpplink predict --schema examples/sample_schema.json --model model.json \
                        --out predictions/ --threshold 20 examples/sample.parquet
```

```text
Candidates     116,939,057
Predictions    169,026  (0.14% of candidates)
Elapsed        5.6 s  (20920301 candidates/s)

Zone           Candidate pairs       Share        Patterns
----------------------------------------------------------
skipped            116,762,308      99.85%               -
drop                     7,723       0.01%          90,019
check                        0       0.00%          10,701
emit                   169,026       0.14%          19,280
----------------------------------------------------------
```

One shard file per thread, nothing held in memory. Name a `.parquet` or `.csv` file in `--out`
instead of a directory and the run ends with a single file, merged from those shards; `cluster`
takes either. **99.85% of candidates never had a string metric run on them**: the pair-global ceiling grants every comparison the best level its cheap
bounds still admit and drops the pair if even that sum cannot clear the threshold. See
[`predict`](commands/predict.md).

### 7. Join the predictions into clusters

```sh
./build/cpplink cluster --schema examples/sample_schema.json --predictions predictions/ \
                        --out clusters.csv --truth examples/sample.truth.csv \
                        examples/sample.parquet
```

```text
122,171 clusters of two or more, covering 265,233 records (14.74%)
  precision  1.0000
  recall     0.9949
  f1          0.9974
```

Edges carry their weight, so re-clustering at a higher `--threshold` is a re-read and no
re-scoring. See [`cluster`](commands/cluster.md).

### 8. Ask the model about one record

Everything above builds a model of what a matching pair looks like. That model is also a search
index: [`search`](commands/search.md) takes a query record and returns the records scoring
highest against it.

```sh
./build/cpplink search --schema examples/sample_schema.json --model model.json \
                       examples/sample.parquet -k 5 \
                       --field last_name=chirdloackki --field dob=1979-08-06 \
                       --field postcode=4508 --expected-matches 1
```

```text
Records        1,800,000
Comparisons    2 tabulated over their dictionary, 1 evaluated per row, 6 constant because the query is missing the column
Values walked  182,320
Rows scored    436 of 1,800,000 exactly; the rest were dropped on the bracket
Prior          -20.780 bits  (the model's lambda says -23.475)
Elapsed        0.0080 s walking the dictionaries, 0.0762 s over the rows on 1 thread

Record                          weight     posterior
r0                              27.785      1.000000
r259138                         -3.045      0.108036
r1482739                        -3.045      0.108036
r632071                         -7.288      0.006357
r1048594                        -7.288      0.006357
```

That is one record of the sample asked for as a query, and it comes back at the top with 27.8
bits while the runners-up sit below even odds. The three counts on the `Comparisons` line are
the three things a comparison can cost: **constant** is free, the query not carrying the column;
**tabulated** is one walk over that column's dictionary and then a byte read per record;
**evaluated** is the pair path's own evaluation per record, which a date or a coordinate pair
needs because it is not a function of a single value id.

No blocking and no candidate pairs: the query is compared against every record, which is
affordable because interning makes a string level a question about a *value*. Every string
metric in the query runs once per **distinct value** of a column rather than once per record,
after which a record costs a load and a compare. A column the query does not name is *missing*
and costs nothing at all.

The answer is exact with respect to the model — the same records, in the same order, as scoring
the query against every record one at a time — because the ranking is on the same admissible
bracket `predict` prunes with.

`--expected-matches 1` is the prior stated as the question a *search* asks. λ is the match rate
over the pair space, which is what deduplication faces; a query against N records asks something
else, and the two differ by orders of magnitude. `--explain` prints the same waterfall
[`explain`](commands/explain.md) does, for each hit.

## Where the time goes

On this run, at 1.8M rows:

| Stage | Wall |
| --- | ---: |
| Load and intern (single-threaded) | 2.6 s |
| `explain-blocking`, exact costing | < 1 s |
| `estimate`, four sessions plus `u` | ~7 s |
| `predict`, 117M candidates on 8 threads | 5.6 s |
| `cluster`, 169k predictions past a union–find | 0.02 s |

Comparison throughput in `predict` is still the binding constraint of the whole pipeline, and
blocking is about 240× cheaper than the comparison it feeds. Making blocking faster would
be effort wasted; see [Blocking](blocking.md#enumeration-is-not-the-bottleneck).

The `predict` figure is what it is because the ceiling refuses 99.85% of candidates before any
string metric runs. Its saving is a function of the threshold, so a run at 0 bits pays much
closer to full price.

## Next

Read [Configuration](reference/schema.md) to point cpplink at your own data — the schema
file is where columns, comparison levels and blocking sources are all declared.
