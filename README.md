# cpplink

[![CI](https://github.com/4ment/cpplink/actions/workflows/ci.yml/badge.svg)](https://github.com/4ment/cpplink/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-4ment.github.io%2Fcpplink-blue.svg)](https://4ment.github.io/cpplink/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**cpplink** is a fast, streaming probabilistic record linker in C++.
It implements the Fellegi-Sunter model, with EM parameter estimation, term-frequency adjustments and prediction, without ever materializing a table of candidate pairs.

This project is heavily inspired by [splink], and reuses many of the ideas and design choices that make Splink such a useful record linkage tool.
Splink was also an important part of how I learned and explored the record linkage problem while developing cpplink, so this project is in large part a didactic exercise as well as a practical tool.
Kudos to the Splink developers for building such a great resource.

The main motivation for cpplink was practical: I wanted to run probabilistic record linkage on datasets that did not fit comfortably in memory on a laptop.
Splink's architecture relies on SQL engines to construct and process intermediate candidate-pair tables, which can become expensive in memory and storage as datasets grow.
Rather than materializing those pairs, cpplink generates candidates as a stream and immediately reduces each pair to its Fellegi-Sunter agreement pattern.

The result is essentially a memory-friendly, C++ implementation of a Splink-style record linkage workflow, without requiring a SQL database or materializing a table of candidate pairs.

## How it works

A pair enters the Fellegi-Sunter likelihood only through its agreement pattern γ, the vector of comparison-level indices.
Two pairs with the same γ are indistinguishable to the model, so estimation needs only `count[γ]`, a histogram bounded by the number of *distinct* patterns rather than the number of pairs.
On the 20M-record deduplication below:

|  | pair table | pattern histogram |
|---|---:|---:|
| Candidate pairs | 1.01×10¹⁰ | 1.01×10¹⁰ |
| Stored rows | 1.01×10¹⁰ | ~10⁵ |
| Resident memory | 121 GB *(at 12 bytes a pair)* | 4.49 GB *(measured)* |
| Cost of one EM re-fit | re-read everything | < 0.1 s |

Blocking is a lazy iterator rather than a join, candidate pairs are deduplicated across sources by a cheap predicate instead of a global `DISTINCT`, and term-frequency adjustment is made affordable by admissible score bounds that let most patterns skip the TF tables entirely.
Blocking is also automatic: rather than hand-written rules, candidates come from agreement on *rare* values, which the term-frequency tables the model already needs supply for free, supplemented by MinHash LSH and sorted-neighbourhood passes, and each source's contribution is measured rather than argued.

The [documentation](https://4ment.github.io/cpplink/) has the model, the blocking sources, a page per command and the schema reference; [Prior art](https://4ment.github.io/cpplink/prior-art/) says which ideas are borrowed and which are this project's.

## Performance

The following benchmark was run on a MacBook Air (Apple M1, 16 GB RAM).
The design target is **20M records**, and the whole pipeline has been run there: `estimate` → `predict` → `cluster` over 20,000,000 synthetic records and **10,090,976,646 candidate pairs** in **38.1 minutes on a single thread**, at **4.49 GB peak resident memory**, with no scratch file and F1 0.9946 against the planted duplicates.
The two figures below are single-threaded sweeps from 250k to 20M records on one machine, drawn against candidate pairs rather than records because that is the axis on which two blocking plans are comparable: time is linear in pairs, while memory tracks records, since nothing in the pipeline holds a row per pair.
Splink is drawn beside it on a shared schema, the intersection of the levels and blocking rules both tools express identically, which is the only way to give two tools the same model; cpplink is run twice, once on each schema, so that line has a like-for-like partner.

![Wall time against candidate pairs, one thread](docs/img/scale-time.svg)

![Peak memory against candidate pairs, one thread](docs/img/scale-memory.svg)

Measured against Splink on the same data, the same schema and the same machine, [`bench/scale/`](bench/scale/) runs 4M records over 1.44bn candidate pairs in **89.2 s at 1.16 GB resident with no scratch file**.
At 1M records, where both tools finish, cpplink is **4.4x** faster end to end than Splink's best configuration here and uses 8.9x less memory, for the same quality (F1 0.9974 against 0.9961).
Splink did not finish at 2M on this machine, and what it ran out of was temp space rather than memory.
On a link between two tables, splink's own [transactions example](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/transactions.html) run with identical candidates and the same two EM sessions, cpplink scores F1 0.7741 against splink's 0.7730, agreeing to the pair at every threshold from 0.9 up, in 1.7 s and 95 MiB against 16.7 s and 1.2 GiB; see [`bench/README.md`](bench/README.md#linking-two-tables-the-transactions-benchmark).

Further information about the benchmark, including precision, recall, and F1 score, can be found in [`bench/scale/`](bench/scale/README.md).

## Quick start

cpplink is told what is in the data with a JSON schema: the columns, a comparison per field as an ordered ladder of levels where the first hit wins, and the blocking sources.
[examples/sample_schema.json](examples/sample_schema.json) is a complete one; the shape is:

```json
{
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "last_name",  "type": "string"},
    {"name": "dob",        "type": "date"},
    {"name": "email",      "type": "string"}
  ],
  "comparisons": [
    {"name": "first_name", "columns": ["first_name"], "term_frequency": true, "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "levenshtein", "threshold": 1}, {"type": "else"}]},
    {"name": "last_name", "columns": ["last_name"], "term_frequency": true, "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "jaro_winkler", "threshold": 0.92}, {"type": "else"}]},
    {"name": "dob", "columns": ["dob"], "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "date_within", "threshold": 2}, {"type": "else"}]},
    {"name": "email", "columns": ["email"], "term_frequency": true, "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "rare_value", "column": "last_name", "max_frequency": 100}
  ]
}
```

`cpplink init` drafts one from a parquet file's column names and types, and the pipeline is then four commands:

```sh
cpplink gen-sample --out sample.parquet --rows 1000000 --truth sample.truth.csv   # or your own parquet file
cpplink init --out schema.json sample.parquet
cpplink estimate --schema schema.json --out model.json sample.parquet
cpplink predict --schema schema.json --model model.json --out predictions.parquet --threshold 20 sample.parquet
cpplink cluster --schema schema.json --predictions predictions.parquet --out clusters.parquet --truth sample.truth.csv sample.parquet
```

Two parquet files instead of one links them, scoring only the pairs with one row in each file.
The same pipeline from Python:

```python
import cpplink

linker = cpplink.Linker("schema.json", "sample.parquet")     # or a pandas frame
model, report = linker.estimate(out="model.json")
predictions = linker.predict(model, threshold=20)            # a pandas frame
clusters = linker.cluster(truth="sample.truth.csv")          # another
print(linker.last_cluster.quality)
```

Between `init` and `estimate` sit the diagnostics that cost seconds and decide the quality of the result: `profile` says what each column can be worth before any model exists, `levels` places the fuzzy thresholds from the data, `explain-blocking` prices every source without enumerating a pair, and `recall` measures what blocking reaches.
See [Getting started](https://4ment.github.io/cpplink/getting-started/) for the whole pipeline with its output explained, [Commands](https://4ment.github.io/cpplink/commands/) for every command at a glance, the [schema reference](https://4ment.github.io/cpplink/reference/schema/) for every field, and [From Python](https://4ment.github.io/cpplink/python/) for the package.

## Installation

cpplink needs CMake 3.15 or higher, a C++17 compiler, Arrow C++ and [nlohmann_json](https://github.com/nlohmann/json).
Everything but the compiler is declared in [environment.yml](environment.yml):

```sh
conda env create -f environment.yml
conda activate cpplink
cmake -S . -B build
cmake --build build
./build/cpplink --version
```

The Python package builds into the same environment, against the Arrow it holds:

```sh
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
```

Outside conda it builds with no Arrow at all, `pip install . -Ccmake.define.CPPLINK_WITH_ARROW=OFF`, and reads and writes parquet through pandas instead; [From Python](https://4ment.github.io/cpplink/python/) says what that build leaves out.

Building with the tests, the sanitizer presets, the lint tools and the documentation are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


[splink]: https://moj-analytical-services.github.io/splink/
