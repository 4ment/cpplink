# cpplink

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**cpplink** is a fast, streaming probabilistic record linker in C++. It implements the
Fellegi–Sunter model — EM parameter estimation, term-frequency adjustments, and prediction
— without ever materializing a table of candidate pairs.

It exists because tools like [splink](https://moj-analytical-services.github.io/splink/)
build the intermediate pair table as a real relation, which does not fit in memory at tens
of millions of records. cpplink folds pairs into a histogram of agreement patterns as they
are generated and discards them, so peak memory is set by the number of *records*, not the
number of *pairs*.

> **Status: phase 0 complete.** The record store, parquet loader and `inspect` command
> are in; comparisons, blocking, estimation and scoring are not.

## Approach

A pair enters the Fellegi–Sunter likelihood only through its agreement pattern γ — the
vector of comparison-level indices. Two pairs with the same γ are indistinguishable to the
model, so estimation needs only `count[γ]`, a histogram bounded by the number of *distinct*
patterns rather than the number of pairs. On a 18M-record deduplication:

|  | pair table | pattern histogram |
|---|---:|---:|
| Candidate pairs | 5×10⁹ | 5×10⁹ |
| Stored rows | 5×10⁹ | ~10⁵ |
| Resident memory | ~130 GB | 3.1 GB *(measured)* |
| Cost of one EM re-fit | re-read everything | < 0.1 s |

Blocking is a lazy iterator rather than a join, candidate pairs are deduplicated across
sources by a cheap predicate instead of a global `DISTINCT`, and term-frequency adjustment
is made affordable by admissible score bounds that let most patterns skip the TF tables
entirely.

Blocking is also **automatic**. Rather than hand-written rules, candidates come from the
term-frequency tables the model already needs — pairs are generated from agreement on
*rare* values, which is exactly the high-evidence event the model scores — supplemented by
MinHash LSH and sorted-neighbourhood passes. Sources that select on a whole record rather
than a column (an ANN index, for instance) are used for prediction only, because they break
the conditional-independence argument that makes EM's parameter estimates unbiased.

## Usage

cpplink is told what is in the data with a JSON schema — see
[examples/sample_schema.json](examples/sample_schema.json):

```json
{
  "unique_id": "id",
  "columns": [
    {"name": "last_name",      "type": "string"},
    {"name": "dob",            "type": "date"},
    {"name": "latitude",       "type": "double"},
    {"name": "address_tokens", "type": "string_list"}
  ]
}
```

`string` columns are interned to dense `uint32` ids, `string_list` columns are stored as
CSR with each row sorted, `date` is days since the epoch, and `double` is stored as-is.
Only the first three carry term frequencies: exact agreement between two doubles is not a
discrete event worth counting, so a `double` column cannot drive rare-value blocking.

```sh
# Report cardinality, null rates and the memory each structure costs
cpplink inspect --schema examples/sample_schema.json data.parquet

# Write a sample file with realistic cardinalities and planted duplicates
cpplink gen-sample --out sample.parquet --rows 18000000 --truth sample.truth.csv
```

`gen-sample` exists because the memory claims above are only worth making if they are
measured. It plants corrupted copies of earlier rows and records them, so the file also
serves as ground truth for the recall harness in a later phase.

## Planned features

- Parquet input with a JSON schema, including list-valued columns *(done)*
- Fellegi–Sunter model with configurable comparisons and ordered comparison levels
- EM estimation of `m` and `λ`; exact closed-form `u` for exact-match levels
- Term-frequency adjustments, with admissible bounds for pruning
- Automatic blocking: rare-value inverted index, MinHash LSH, sorted neighbourhood,
  optional ANN — with exact candidate-count reporting and recall measurement before any run
- Multicore, shared-memory parallelism
- Connected-component clustering of the scored edges
- Deduplication first; record linkage across datasets through the same interfaces

## Getting Started

### Prerequisites

- CMake 3.15 or higher
- A C++17 compatible compiler
- GoogleTest (for the test suite)

The dependencies are declared in [environment.yml](environment.yml):

```sh
conda env create -f environment.yml
conda activate cpplink
```

### Building

```sh
cmake -S . -B build
cmake --build build
```

### Testing

```sh
cmake -S . -B build -DBUILD_TESTING=on -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
ctest --test-dir build
```

## Code Style

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
with minor customizations (4-space indent, 90-column limit).

Tools used:

- clang-format for formatting — `cmake --build build --target format`
- cpplint for static analysis — `cpplint --recursive src tests`

## License

This project is licensed under the MIT License.
