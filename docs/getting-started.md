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

## The whole pipeline in seven commands

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
Load time    2.8 s  (648,979 rows/s)

Column            Type                Distinct      Null   Top val Mean len        Rare pairs
---------------------------------------------------------------------------------------------
first_name        string                23,060     0.00%     2.27%        -            52,729
last_name         string               173,347     0.00%     0.20%        -        16,675,804
dob               date                  20,821     0.00%     0.01%        -        68,631,954
...
Resident total                        333.8 MB   194 bytes per record
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
Sum over sources                                  130,818,941
Union, deduplicated                               116,936,544
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
Union                               131623    91.58%

12107 known pairs are reachable by no source.
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
                        --out edges/ --threshold 20 examples/sample.parquet
```

```text
Candidates     116,936,544
Edges          142,541  (0.12% of candidates)
Elapsed        36.0 s  (3246923 candidates/s)
```

One shard file per thread, nothing held in memory. See [`predict`](commands/predict.md).

### 7. Join the edges into clusters

```sh
./build/cpplink cluster --schema examples/sample_schema.json --edges edges/ \
                        --out clusters.csv --truth examples/sample.truth.csv \
                        examples/sample.parquet
```

```text
122,778 clusters of two or more, covering 255,218 records (14.18%)
  precision  0.9211
  recall     0.9158
  f1          0.9184
```

Edges carry their weight, so re-clustering at a higher `--threshold` is a re-read and no
re-scoring. See [`cluster`](commands/cluster.md).

## Where the time goes

On this run, at 1.8M rows:

| Stage | Wall |
| --- | ---: |
| Load and intern (single-threaded) | 2.8 s |
| `explain-blocking`, exact costing | < 1 s |
| `estimate`, four sessions plus `u` | ~7 s |
| `predict`, 117M candidates on 8 threads | 36.0 s |
| `cluster`, 142k edges past a union–find | 0.02 s |

Comparison throughput in `predict` is the binding constraint of the whole pipeline, and
blocking is about 240× cheaper than the comparison it feeds. Making blocking faster would
be effort wasted; see [Blocking](blocking.md#enumeration-is-not-the-bottleneck).

## Next

Read [Configuration](reference/schema.md) to point cpplink at your own data — the schema
file is where columns, comparison levels and blocking sources are all declared.
